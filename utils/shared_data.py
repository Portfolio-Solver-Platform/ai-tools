from pathlib import Path
import numpy as np
from utils.raw_data_to_training_data.instance_benchmark_to_training_data import convert


def get_24_25_data():
    root = Path(__file__).resolve().parent.parent
    benchmark = [
        root / "portfolio-benchmarks/24-25/cpsat8-benchmark.csv",
        root / "portfolio-benchmarks/24-25/best-static-benchmark-restart.csv",
    ]
    instances = Path("/home/sofus/speciale/ai/24_25_instances")
    return convert(benchmark, instances)


def get_23_data():
    root = Path(__file__).resolve().parent.parent
    benchmark = [
        root / "portfolio-benchmarks/2023/cpsat8-benchmark-2023.csv",
        root / "portfolio-benchmarks/2023/best-static-benchmark-2023-restart.csv",
    ]
    instances = Path("/home/sofus/speciale/ai/23_instances")
    return convert(benchmark, instances)


def get_mznc_data():
    root = Path(__file__).resolve().parent.parent
    benchmark = [
        root / "portfolio-benchmarks/24-25/cpsat8-benchmark.csv",
        root / "portfolio-benchmarks/24-25/best-static-benchmark-restart.csv",
    ]
    instances = [
        Path("/home/sofus/speciale/ai/data/mznc2023_probs"),
        Path("/home/sofus/speciale/ai/data/mznc2024_probs"),
        Path("/home/sofus/speciale/ai/data/mznc2025_probs"),
    ]
    return convert(benchmark, instances)


def prepare_labels(Y):
    max_real_time = np.nanmax(Y)
    timeout_penalty = max_real_time * 2
    Y_filled = np.nan_to_num(Y, nan=timeout_penalty)
    y_labels = np.argmin(Y_filled, axis=1)
    Y_eval = np.nan_to_num(Y, nan=max_real_time)
    return y_labels, Y_eval
