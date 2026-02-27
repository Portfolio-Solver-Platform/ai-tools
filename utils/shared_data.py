from pathlib import Path
import numpy as np
from utils.raw_data_to_training_data.instance_benchmark_to_training_data import convert


def get_24_25_data():
    benchmark = [
        Path("/home/sofus/speciale/ai/benchmarks/24-25/cpsat8-benchmark.csv"),
        Path("/home/sofus/speciale/ai/benchmarks/24-25/best-static-benchmark-restart.csv"),
    ]
    instances = Path("/home/sofus/speciale/ai/24_25_instances")
    return convert(benchmark, instances)


def get_23_data():
    benchmark = [
        Path("/home/sofus/speciale/ai/benchmarks/2023/cpsat8-benchmark-2023.csv"),
        Path("/home/sofus/speciale/ai/benchmarks/2023/best-static-benchmark-2023.csv"),
    ]
    instances = Path("/home/sofus/speciale/ai/23_instances")
    return convert(benchmark, instances)


def prepare_labels(Y):
    """Convert raw Y times into training labels and eval matrix.

    Returns (y_labels, Y_eval) where:
      - y_labels: argmin solver per instance (NaN treated as 2x max penalty)
      - Y_eval: Y with NaN replaced by max observed time
    """
    max_real_time = np.nanmax(Y)
    timeout_penalty = max_real_time * 2
    Y_filled = np.nan_to_num(Y, nan=timeout_penalty)
    y_labels = np.argmin(Y_filled, axis=1)
    Y_eval = np.nan_to_num(Y, nan=max_real_time)
    return y_labels, Y_eval
