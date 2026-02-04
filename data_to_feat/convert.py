from pathlib import Path
import json
from typing import NamedTuple 
from file_read_backwards import FileReadBackwards
import numpy as np
import pickle 

ACCEPTED_STATUS = ["ALL_SOLUTIONS", "OPTIMAL_SOLUTION", "UNSATISFIABLE", "SATISFIED"]
UNEXPECTED_STATUS = ["UNBOUNDED", "UNSAT_OR_UNBOUNDED"]
NAN_STATUS = ["ERROR", "UNKNOWN"]

SOLVER_ORDER = ["choco", "chuffed", "cp-sat", "gecode", "Huub", "Picat"]

def extract_time(folder_path):
    out_files = sorted(list(folder_path.glob('*.out')))
    times_dict = dict()
    # insert_time(problems, "../mzn_challenge_results/accap_accap_a3_f20_t10-sep-choco-sep-1.out", times_dict, 0)

    for file in out_files:
        with FileReadBackwards(file, encoding="utf-8") as frb:
            i = 0
            for line in frb:
                i += 1
                data = json.loads(line)
                data_type = data.get("type") 
                if data_type == "status":
                    status = data.get("status")
                    if status in ACCEPTED_STATUS:
                        time = data.get("time")
                        insert_time(file, times_dict, time)
                    elif status in UNEXPECTED_STATUS:   
                        print(f"Unexpected status showed up: {status} in file {file}. Please make sure the code works for this, as it has not been tested")
                        time = data.get("time")
                        insert_time(file, times_dict, time)
                    elif status in NAN_STATUS:
                        time = np.nan
                        insert_time(file, times_dict, time)
                    break 
                elif data_type == "solution":
                    time = np.nan
                    insert_time(file, times_dict, time)
                    break
                elif data_type == "error":
                    time = np.nan
                    insert_time(file, times_dict, time)
                    break

            if i == 0:
                time = np.nan
                insert_time(file, times_dict, time)

    return times_dict
            
def insert_time(file_path, times_dict, time):
    try:
        file_path = file_path.name.split("/")[-1]
        file_path_split = file_path.split("-sep-")
        problem_instance = file_path_split[0]
        solver = file_path_split[1]
        cores = int(file_path_split[2].split(".out")[0])
        if cores != 1: # TODO: experiment with how we can use times that use multiple cores.
            return
        if solver == "CPLEX" or solver == "portfolio":
            return
        solver_idx = SOLVER_ORDER.index(solver)
        insert_or_update(times_dict, problem_instance, solver_idx, time)
    except:
        print(f"Failed to read the this file: {file_path}")
    
def insert_or_update(my_dict, key, target_idx, number):
    if key not in my_dict:
        new_list_structure = [[] for _ in range(len(SOLVER_ORDER))]
        my_dict[key] = new_list_structure
    my_dict[key][target_idx].append(number)
    
def get_average_time(folder_path):
    times_dict = extract_time(folder_path)
    average_times = dict()
    for key, solvers_times in times_dict.items():
        times = np.full(len(SOLVER_ORDER), np.nan, dtype=np.float64)
        for i, solver_times in enumerate(solvers_times):
            average = np.nanmean(solver_times)
            times[i] = average
        if np.any(~np.isnan(times)): # check if at least one solver has a time
            average_times[key] = times
    return average_times

def to_numpy(folder_path, features_path):
    average_times = get_average_time(folder_path)
    features = None
    with open(features_path, 'rb') as f:  
        features = pickle.load(f)
        
    common_keys = average_times.keys() & features.keys()

    sorted_keys = sorted(common_keys)

    # Y = [average_times[k] for k in sorted_keys]
    # X = [features[k] for k in sorted_keys]

    valid_keys = [k for k in sorted_keys if features[k] is not None]                                                  
                                                                                                                      
    Y = [average_times[k] for k in valid_keys]                                                                        
    X = [features[k] for k in valid_keys]        
    Y = np.array(Y)
    X = np.vstack(X)
    
    np.savez('../data/training_data.npz',X=X, Y=Y)
            



if __name__ == "__main__":
    folder_path = Path('../mzn_challenge_results')
    feature_path = Path('features_data.pkl')
    to_numpy(folder_path, feature_path)