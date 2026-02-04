
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import xgboost2 as xgb
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor


print("EXPERIMENT 1")

data = np.load('../data/training_data.npz')
X = data["X"]
Y = data["Y"]
y_labels = np.argmin(Y, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"KNN Accuracy: {accuracy:.2f}")


print("EXPERIMENT 2")

knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X, y_labels)

predictions = knn.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


print("EXPERIMENT 3 - XGBOOST")
xgb_estimator = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    n_jobs=-1  # Use all CPU cores
)

# Wraps it to handle [t1, t2, t3, t4, t5, t6] output
model = MultiOutputRegressor(xgb_estimator)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

pred_best_solver_indices = np.argmin(predictions, axis=1)
true_best_solver_indices = np.argmin(y_test, axis=1)

accuracy = np.mean(pred_best_solver_indices == true_best_solver_indices)
print(f"XGBoost Selector Accuracy: {accuracy * 100:.2f}%")
chosen_times = y_test[np.arange(len(y_test)), pred_best_solver_indices]
# We pick the actual runtime of the ideal solver
ideal_times = np.min(y_test, axis=1)
avg_time_loss = np.mean(chosen_times - ideal_times)

print(f"Average Time Lost per Problem: {avg_time_loss:.4f} seconds")

print("EXPERIMENT 4 - Gaussion process")

gps = []
print("Training GPs...")
for i in range(3):
    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
    
    # Train on LOG time
    gp.fit(X_train, np.log(y_train[:, i]))
    gps.append(gp)

# --- 3. Predict on Test Set ---
predicted_means_log = []
predicted_stds_log = []

for gp in gps:
    # Get mean and uncertainty (std)
    m, s = gp.predict(X_test, return_std=True)
    predicted_means_log.append(m)
    predicted_stds_log.append(s)

# Convert list to array (Shape: Solvers x Samples) -> Transpose to (Samples x Solvers)
pred_log_times = np.array(predicted_means_log).T
pred_uncertainties = np.array(predicted_stds_log).T

# Convert log-predictions back to seconds (just for our view, argmin is same for log or linear)
pred_times_seconds = np.exp(pred_log_times)


my_choices = np.argmin(pred_times_seconds, axis=1)

true_winners = np.argmin(y_test, axis=1)

# Calculate Classification Accuracy
correct_picks = np.sum(my_choices == true_winners)
total_samples = len(y_test)
accuracy = correct_picks / total_samples

print(f"\n--- Results ---")
print(f"Classification Accuracy: {accuracy * 100:.2f}% (How often we picked the absolute best)")
my_runtimes = y_test[np.arange(total_samples), my_choices]
# The time of the perfect solver
best_possible_runtimes = y_test[np.arange(total_samples), true_winners]

avg_time_loss = np.mean(my_runtimes - best_possible_runtimes)
print(f"Average Time Lost: {avg_time_loss:.4f} seconds per problem")