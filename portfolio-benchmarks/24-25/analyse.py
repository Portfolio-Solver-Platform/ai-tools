import pandas as pd
import numpy as np

df_no_restart = pd.read_csv("best-static-benchmark-no-restart.csv")
df_restart = pd.read_csv("best-static-benchmark-restart.csv")
df_cpsat8 = pd.read_csv("cpsat8-benchmark.csv")

no_restart_sum = np.sum(df_no_restart["time_ms"])
restart_sum = np.sum(df_restart["time_ms"])
cpsat8_sum = np.sum(df_cpsat8["time_ms"])

print("no restart time:", no_restart_sum)
print("restart time:", restart_sum)
print("cpsat8 time:", cpsat8_sum)

no_restart_optimal_count = len(df_no_restart[df_no_restart["optimal"] == "Optimal"])
restart_optimal_count = len(df_restart[df_restart["optimal"] == "Optimal"])
cpsat8_optimal_count = len(df_cpsat8[df_cpsat8["optimal"] == "Optimal"])

print("no restart optimal count:", no_restart_optimal_count)
print("restart optimal count:", restart_optimal_count)
print("cpsat8 optimal count:", cpsat8_optimal_count)

