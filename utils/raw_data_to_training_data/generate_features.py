import os
import pickle
import subprocess
import tempfile
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def generate_features_for_everything(problems_path, save_dict, n_workers=None):
    jobs = []
    for sub_folder in sorted(os.listdir(problems_path)):
        sub_folder_path = os.path.join(problems_path, sub_folder)
        if not os.path.isdir(sub_folder_path):
            continue

        models = sorted(f for f in os.listdir(sub_folder_path) if f.endswith('.mzn'))
        instances = sorted(
            os.path.relpath(os.path.join(root, f), sub_folder_path)
            for root, _, files in os.walk(sub_folder_path)
            for f in files
            if f.endswith('.dzn') or f.endswith('.json')
        )

        if not instances:
            for model in models:
                model_path = os.path.join(sub_folder_path, model)
                key = ".".join(model.split(".")[:-1]) + "_"
                jobs.append((model_path, None, key))
        else:
            for model in models:
                for instance in instances:
                    model_path = os.path.join(sub_folder_path, model)
                    instance_path = os.path.join(sub_folder_path, instance)
                    instance_name = os.path.basename(instance)
                    key = ".".join(model.split(".")[:-1]) + "_" + ".".join(instance_name.split(".")[:-1])
                    jobs.append((model_path, instance_path, key))

    if not jobs:
        print("No instances found")
        return

    if n_workers is None:
        n_workers = min(cpu_count(), len(jobs))

    print(f"Generating features for {len(jobs)} instances with {n_workers} workers")

    with Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(_generate_one, jobs),
            total=len(jobs),
        ))

    for key, feature in results:
        if feature is not None:
            save_dict[key] = feature

    print(f"Generated {sum(1 for _, f in results if f is not None)}/{len(jobs)} features")


def _generate_one(args):
    model, instance, key = args
    feature = generate_features(model, instance)
    return key, feature


def generate_features(model: str, instance: str | None) -> np.ndarray:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fzn_path = os.path.join(tmpdir, "model.fzn")
            ozn_path = os.path.join(tmpdir, "output.ozn")

            cmd = ['minizinc', model]
            if instance:
                cmd.append(instance)
            cmd.extend(['--solver', 'gecode', '-c', '--fzn', fzn_path, '--ozn', ozn_path])
            if not instance:
                pass
            else:
                cmd.extend(['--output-mode', 'json'])

            p = subprocess.run(cmd, capture_output=True, timeout=300)
            if p.returncode != 0:
                return None

            out = subprocess.run(['fzn2feat', fzn_path], capture_output=True, timeout=60)
            stdout = out.stdout.decode()
            if stdout == '' or 'nan' in stdout or 'inf' in stdout:
                return None
            return np.array([[float(f) for f in stdout.split(',')]])

    except Exception:
        return None


if __name__ == "__main__":
    import sys
    problems_path = sys.argv[1] if len(sys.argv) > 1 else '../../24_25_instances'
    save_dict = {}
    generate_features_for_everything(problems_path, save_dict)
    print(len(save_dict))
    with open('features_data.pkl', 'wb') as f:
        pickle.dump(save_dict, f)
