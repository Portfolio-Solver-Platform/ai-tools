import pandas as pd
import numpy as np

svm_df = pd.read_csv("svm-2023.csv")
cpsat8_df = pd.read_csv("cpsat8-benchmark-2023.csv")
best_static_df = pd.read_csv("best-static-benchmark-2023.csv")

restart_sum = np.sum(svm_df["time_ms"])
cpsat8_sum = np.sum(cpsat8_df["time_ms"])
best_static_sum = np.sum(best_static_df["time_ms"])

print("svm time:", restart_sum)
print("cpsat8 time:", cpsat8_sum)
print("best_static time:", best_static_sum)

restart_optimal_count = len(svm_df[svm_df["optimal"] == "Optimal"])
cpsat8_optimal_count = len(cpsat8_df[cpsat8_df["optimal"] == "Optimal"])
best_static_optimal_count = len(best_static_df[best_static_df["optimal"] == "Optimal"])

print("svm optimal count:", restart_optimal_count)
print("cpsat8 optimal count:", cpsat8_optimal_count)
print("best_static optimal count:", best_static_optimal_count)

