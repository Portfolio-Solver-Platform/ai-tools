from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import joblib
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut



data = np.load('../data/training_data.npz')
X = data["X"]
Y = data["Y"]

max_real_time = np.nanmax(Y)
timeout_penalty = max_real_time * 1.1 # experiment with scalar


Y_filled = np.nan_to_num(Y, nan=timeout_penalty)
Y_log = np.log(Y_filled)
y_labels = np.argmin(Y_filled, axis=1)
print(y_labels)


indices = np.arange(len(X))
X_train, X_test, y_train, y_test, y_times_train, y_times_test = train_test_split(
    X, y_labels, Y_filled, test_size=0.05, stratify=y_labels,
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

knn_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_neighbors=5))
])


knn_pipe.fit(X_train, y_train)
probs = knn_pipe.predict_proba(X_test)

total_time = 0
best_possible_time = 0
correct_solver = 0
n_test_samples = len(X_test)
classes = knn_pipe.classes_ 

for i in range(n_test_samples):
    active_solvers_idx = classes[probs[i] > 0]
    actual_times = y_times_test[i][active_solvers_idx]
    print(len(actual_times))
    # The portfolio time is the best (minimum) time among the active solvers
    best_solver_idx = np.argmin(y_times_test[i])
    best_possible_time += y_times_test[i][best_solver_idx]
    if best_solver_idx in active_solvers_idx:
        correct_solver += 1
    best_time = np.min(actual_times)
    total_time += best_time
    


avg_time = total_time / n_test_samples
print(f"{correct_solver}/{n_test_samples}")
print(avg_time)
print(total_time)
print(f"Best possible time: {best_possible_time}")

joblib.dump(knn_pipe, 'knn_model.pkl')


