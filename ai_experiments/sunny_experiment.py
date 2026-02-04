import joblib
import numpy as np
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

print("PRE-PROCESSING: LOG-SPACE TRANSFORMATION")

data = np.load('../data/training_data.npz')
X = data["X"]
Y = data["Y"]

# --- 1. HANDLE TIMEOUTS ---
max_real_time = np.nanmax(Y)
# Set penalty slightly higher than the worst valid run (PAR-10 or similar logic)
timeout_penalty = max_real_time * 1.1 
print(f"Max observed runtime: {max_real_time:.2f}s")
print(f"Timeouts (NaN) replaced with: {timeout_penalty:.2f}s")

Y_filled = np.nan_to_num(Y, nan=timeout_penalty)

# --- 2. CREATE LOG TARGETS ---
Y_log = np.log(Y_filled + 1e-6)

# --- 3. CREATE LABELS ---
y_labels = np.argmin(Y_filled, axis=1)

# --- 4. SCALE INPUTS ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 5. SPLIT INDICES (Keeps everything aligned) ---
indices = np.arange(len(X))
X_train, X_test, idx_train, idx_test = train_test_split(
    X_scaled, indices, test_size=0.01, #random_state=42
)

# Map indices back to data
y_labels_train = y_labels[idx_train]
y_labels_test  = y_labels[idx_test]
Y_train_log    = Y_log[idx_train]
Y_test_seconds = Y_filled[idx_test]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

print("\n--- EXPERIMENT 1: KNN (k=5) ---")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_labels_train)
acc = accuracy_score(y_labels_test, knn.predict(X_test))
print(f"KNN (5) Accuracy: {acc:.2f}")


print("\n--- EXPERIMENT 2: KNN (k=20) ---")
# Note: With only 94 samples, k=20 averages ~20% of the dataset. 
# This is likely too "smooth" and might underperform k=5.
knn20 = KNeighborsClassifier(n_neighbors=20)
knn20.fit(X_train, y_labels_train)
acc = accuracy_score(y_labels_test, knn20.predict(X_test))
print(f"KNN (20) Accuracy: {acc:.2f}")


print("\n--- EXPERIMENT 3: XGBoost (Log-Space Regression) ---")
xgb_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100, 
    learning_rate=0.1,
    max_depth=6, 
    n_jobs=-1
)
# model = MultiOutputRegressor(xgb_estimator)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MultiOutputRegressor(xgb_estimator))
])


# Train on LOG times
model.fit(X_train, Y_train_log)

# Predict (Log space)
pred_log = model.predict(X_test)

# Evaluate
pred_choices = np.argmin(pred_log, axis=1)
true_best = np.argmin(Y_test_seconds, axis=1)

acc = np.mean(pred_choices == true_best)
print(f"XGBoost Accuracy: {acc * 100:.2f}%")

# Time Loss
chosen_times = Y_test_seconds[np.arange(len(Y_test_seconds)), pred_choices]
ideal_times = np.min(Y_test_seconds, axis=1)
print(f"Avg Time Lost: {np.mean(chosen_times - ideal_times):.4f}s")

joblib.dump(model, 'xgboost_multioutput_model.pkl')
print("xgboost saved successfully.")

print("\n--- EXPERIMENT 4: Gaussian Process (Log-Space) ---")
# GP is great for small data (N < 1000)
if len(X_train) > 2000:
    print(f"Subsampling GP training data to 2000...")
    gp_idxs = np.random.choice(len(X_train), 2000, replace=False)
    X_train_gp = X_train[gp_idxs]
    Y_train_gp = Y_train_log[gp_idxs]
else:
    X_train_gp = X_train
    Y_train_gp = Y_train_log

gps = []
num_solvers = Y_log.shape[1]

print(f"Training {num_solvers} GPs on Log-Data...")
for i in range(num_solvers):
    # Normalize Target (Mean 0, Std 1) for better GP convergence
    y_col = Y_train_gp[:, i]
    mean_y, std_y = np.mean(y_col), np.std(y_col)
    y_target_norm = (y_col - mean_y) / (std_y + 1e-9)
    
    # # Lower bound on noise to 1e-5 to fix warnings
    # kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    
    # gp.fit(X_train_gp, y_target_norm)
    # gps.append((gp, mean_y, std_y))
    kernel = C(1.0) * RBF(length_scale=1.0) + \
             WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 10.0))
    
    # n_restarts_optimizer=5 helps it find the best 'smooth' fit rather than a 'jagged' fit
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42)
    
    gp.fit(X_train_gp, y_target_norm)
    gps.append((gp, mean_y, std_y))
    
# Predict
pred_means = []
for gp, mean_y, std_y in gps:
    pred_norm = gp.predict(X_test, return_std=False)
    # De-normalize
    pred_log_scale = pred_norm * std_y + mean_y
    pred_means.append(pred_log_scale)

# Transpose
pred_log_matrix = np.array(pred_means).T

my_choices = np.argmin(pred_log_matrix, axis=1)
acc = np.mean(my_choices == true_best)
print(f"GP Accuracy: {acc * 100:.2f}%")

my_runtimes = Y_test_seconds[np.arange(len(Y_test_seconds)), my_choices]
ideal_runtimes = Y_test_seconds[np.arange(len(Y_test_seconds)), true_best]
print(f"Avg Time Lost: {np.mean(my_runtimes - ideal_runtimes):.4f}s")