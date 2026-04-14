from pathlib import Path
import json
from typing import NamedTuple 
import numpy as np
import pickle 
import pandas as pd
from pathlib import Path

from utils.raw_data_to_training_data.generate_features import generate_features_for_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SAVE_DATA_PATH = PROJECT_ROOT / "data"

ACCEPTED_STATUS = ["ALL_SOLUTIONS", "OPTIMAL_SOLUTION", "UNSATISFIABLE", "SATISFIED"]
UNEXPECTED_STATUS = ["UNBOUNDED", "UNSAT_OR_UNBOUNDED"]
NAN_STATUS = ["ERROR", "UNKNOWN"]


def get_times(benchmarks: list[Path]):
    my_dict = dict()
    
    for i, path in enumerate(benchmarks):
        df = pd.read_csv(path)
    
        for _, (time, instance_name, model) in df[["time_ms", "name", "model"]].iterrows():
            key = f"{model}_" if model == instance_name else f"{model}_{instance_name}"
            insert_or_update(my_dict, key, i, time, len(benchmarks))
    
    return my_dict
    
def insert_or_update(my_dict, key, target_idx, number, n_solvers):
    if key not in my_dict:
        new_list_structure = [[] for _ in range(n_solvers)]
        my_dict[key] = new_list_structure
    my_dict[key][target_idx].append(number)

def load_features(instances_paths: list[Path]) -> dict:
    """Load or generate features from one or more instance directories."""
    all_features = {}
    for instances_path in instances_paths:
        saved_features_path = SAVE_DATA_PATH / f"{instances_path.name}_features.pkl"
        if saved_features_path.is_file():
            with open(saved_features_path, 'rb') as f:
                features = pickle.load(f)
        else:
            features = {}
            generate_features_for_everything(instances_path, features)
            with open(saved_features_path, 'wb') as f:
                pickle.dump(features, f)
        all_features.update(features)
    return all_features


def to_numpy(benchmarks, instances_paths):
    average_times = get_times(benchmarks)
    features = load_features(instances_paths)

    common_keys = average_times.keys() & features.keys()
    print(f"Only in average_times: {average_times.keys() - features.keys()}")
    print(f"Only in features: {features.keys() - average_times.keys()}")

    sorted_keys = sorted(common_keys)
    valid_keys = [k for k in sorted_keys if features[k] is not None]

    Y = [[np.nanmean(solver_times) for solver_times in average_times[k]] for k in valid_keys]
    X = [features[k] for k in valid_keys]
    Y = np.array(Y)
    X = np.vstack(X)

    return X, Y


def convert(benchmarks: list[Path], instances: Path | list[Path]):
    if isinstance(instances, list):
        cache_name = "_".join(sorted(p.name for p in instances))
        instances_paths = instances
    else:
        cache_name = instances.name
        instances_paths = [instances]
    saved_data_path = SAVE_DATA_PATH / f"{cache_name}_training_data.npz"
    if saved_data_path.is_file():
        data = np.load(saved_data_path)
        X = data["X"]
        Y = data["Y"]
        return X, Y
    else:
        X, Y = to_numpy(benchmarks, instances_paths)
        np.savez(saved_data_path, X=X, Y=Y)
        return X, Y
