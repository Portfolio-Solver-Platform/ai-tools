import os
import pickle
import subprocess
from time import time

import numpy as np
from tqdm import tqdm
CACHE = "/tmp"


def generate_features_for_everything(problems_path, save_dict):    
    for sub_folder in tqdm(sorted(os.listdir(problems_path))):
        if not os.path.isdir(os.path.join(problems_path, sub_folder)):
            continue
        # print(sub_folder)
        files:list[str] = os.listdir(os.path.join(problems_path, sub_folder))
        models:list[str] = [file for file in files if file.endswith('.mzn')]
        instances:list[str] = [file for file in files if file.endswith('.dzn') or file.endswith('.json')]
        sub_folder_path = os.path.join(problems_path, sub_folder)

        if len(instances) == 0:
            for model in models:
                model_path = os.path.join(sub_folder_path, model)
                feature = generate_features(model_path, None)
                if feature is not None:
                    save_path = model.split(".")[0] + "_"
                    save_dict[save_path] = feature
        else:
            for model in models:
                for instance in instances:
                    model_path = os.path.join(sub_folder_path, model)
                    instance_path = os.path.join(sub_folder_path, instance)
                    feature = generate_features(model_path, instance_path)
                    if feature is not None:
                        save_path = model.split(".")[0]  + "_" + instance.split(".")[0]
                        save_dict[save_path] = feature
                    
def generate_features(model:str, instance:str|None) -> np.ndarray:
    t = time()
    try:
        if instance:
            p = subprocess.Popen(['minizinc', model, instance, '--solver', 'gecode', '-c', '--fzn', f'{CACHE}/_gecode.fzn', '--ozn', f'{CACHE}/output.ozn', '--output-mode', 'json'], stderr=subprocess.PIPE, stdout=subprocess.PIPE, preexec_fn=os.setsid)
        else:
            p = subprocess.Popen(['minizinc', model, '--solver', 'gecode', '-c', '--fzn', f'{CACHE}/_gecode.fzn', '--ozn', f'{CACHE}/output.ozn'], stderr=subprocess.PIPE, stdout=subprocess.PIPE, preexec_fn=os.setsid)

        out = p.communicate()  # Wait for process to complete
        # Check if process was killed (negative returncode) or failed (non-zero returncode)
        if p.returncode != 0 or p.returncode < 0:
            print(f"failed to extract features, got: {out}")
            return None


        t = time() - t
        out = subprocess.run(['fzn2feat', f'{CACHE}/_gecode.fzn', ], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout = out.stdout.decode()
        if stdout == '':
            print(f"model: {model}, instance: {instance}")
            print(f'{{"type":"info", "info":"no stdout"}}')
            return None
        if 'nan' in stdout or 'inf' in stdout:
            print(f"model: {model}, instance: {instance}")
            print(f'{{"type":"info", "info":"inf or nan in prediction"}}')
            return None
        return np.array([[float(f) for f in stdout.split(',')]])

    except Exception as e:
        print(f'{{"type":"info", "info":"feature extraction error: {e}"}}')
        return None

if __name__ == "__main__":
    save_dict = dict()
    problems_path = '../24_25_instances'
    generate_features_for_everything(problems_path, save_dict)

    with open('features_data.pkl', 'wb') as f:  # 'wb' means write binary
        pickle.dump(save_dict, f)
