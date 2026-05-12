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


def _load_xy(filename):
    path = Path(__file__).resolve().parent.parent / "data" / filename
    data = np.load(path, allow_pickle=True)
    return data["X"], data["Y"], data["meta"]


def get_cpsat8_ek1_data():
    return _load_xy("portfolios_cpsat8_ek1_training_data.npz")


def get_cpsat8_k1_data():
    return _load_xy("portfolios_cpsat8_k1_training_data.npz")


def get_cpsat8_k1_ek1_data():
    return _load_xy("portfolios_cpsat8_k1_ek1_training_data.npz")


def prepare_labels(Y):
    y_labels = np.argmax(Y, axis=1)
    return y_labels, Y
