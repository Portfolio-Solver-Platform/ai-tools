"""
For each combo of open-category solvers (fixed + N extras), train an SVC
to pick the best solver per instance, evaluated by Borda score.

Uses GroupKFold (k=5) by problem so test problems are never seen during training.
Optuna tunes C/gamma per combo.

Usage:
    python train_svc_solvers.py [n_extra]
    python train_svc_solvers.py [n_extra] --no-fixed
"""
import argparse
import csv
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
DATA_DIR = ROOT / "data"

N_FOLDS = 5
N_TRIALS = 50
FIXED = ("cp-sat", 8)


def load_features():
    features = {}
    for pkl in sorted(DATA_DIR.glob("mznc*_features.pkl")):
        with open(pkl, "rb") as f:
            features.update(pickle.load(f))
    return features


def build_data(open_csv, types_csv):
    with open(open_csv) as f:
        rows = list(csv.DictReader(f))

    problem_types = load_problem_types(types_csv)
    open_configs = {(r["solver"], int(r["cores"])) for r in rows if r["open_category"] == "True"}
    scores, configs, instances = borda_scores(rows, problem_types, opponents=open_configs)
    config_idx = {c: i for i, c in enumerate(configs)}

    features = load_features()

    # Build aligned arrays: only instances that have features
    instance_keys_list = []
    instance_problems = []
    instance_idxs = []
    for i, (problem, name) in enumerate(instances):
        model = None
        for r in rows:
            if r["problem"] == problem and r["name"] == name:
                model = r["model"]
                break
        fkey = f"{model}_{name}" if model != name else f"{model}_"
        if fkey in features and features[fkey] is not None:
            instance_keys_list.append(fkey)
            instance_problems.append(problem)
            instance_idxs.append(i)

    X = np.vstack([features[k] for k in instance_keys_list])
    problems = np.array(instance_problems)
    inst_idxs = np.array(instance_idxs)

    # Borda score matrix for open-category solvers
    solver_configs = sorted(open_configs)
    solver_names = [f"{s}({c}c)" for s, c in solver_configs]
    borda_matrix = np.zeros((len(solver_configs), len(inst_idxs)))
    for si, cfg in enumerate(solver_configs):
        ci = config_idx[cfg]
        borda_matrix[si] = scores[ci, inst_idxs]

    return X, borda_matrix, problems, solver_names, solver_configs


def _run_one(args):
    X, borda_matrix, problems, combo_indices, names = args

    combo_borda = borda_matrix[combo_indices]
    labels = np.argmax(combo_borda, axis=0)
    oracle_borda = combo_borda.max(axis=0).sum()

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(X, labels, groups=problems))

    def objective(trial):
        C = trial.suggest_float("C", 0.1, 100, log=True)
        gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)

        total_borda = 0.0
        for train_idx, test_idx in folds:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svc", SVC(kernel="rbf", C=C, gamma=gamma)),
            ])
            pipe.fit(X[train_idx], labels[train_idx])
            preds = pipe.predict(X[test_idx])
            total_borda += sum(combo_borda[preds[i], test_idx[i]] for i in range(len(test_idx)))

        return total_borda

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    return names, study.best_value, oracle_borda, study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_extra", type=int, help="Number of extra solvers beyond the fixed one")
    parser.add_argument("--no-fixed", action="store_true",
                        help="No fixed solver; pick n_extra solvers freely")
    args = parser.parse_args()

    print("Loading data...")
    X, borda_matrix, problems, solver_names, solver_configs = build_data(OPEN_CSV, TYPES_CSV)
    print(f"Instances: {X.shape[0]}, Features: {X.shape[1]}, Solvers: {len(solver_names)}")

    if args.no_fixed:
        k = args.n_extra
        candidate_idxs = list(range(len(solver_names)))
        combos = list(combinations(candidate_idxs, k))
        fixed_label = ""
        print(f"Mode: best {k} from {len(solver_names)} solvers (no fixed)")
    else:
        fixed_cfg = FIXED
        fixed_name = f"{fixed_cfg[0]}({fixed_cfg[1]}c)"
        fixed_idx = solver_names.index(fixed_name)
        fixed_borda = borda_matrix[fixed_idx].sum()
        print(f"Fixed: {fixed_name} (solo Borda: {fixed_borda:.1f})")

        candidate_idxs = [i for i in range(len(solver_names)) if i != fixed_idx]
        combos = list(combinations(candidate_idxs, args.n_extra))
        fixed_label = f"{fixed_name} + "
        print(f"Mode: {fixed_name} + best {args.n_extra} from {len(candidate_idxs)} solvers")

    print(f"Combos: {len(combos)}")

    n_workers = max(1, os.cpu_count() - 1)
    print(f"Workers: {n_workers}")

    tasks = []
    for extra in combos:
        if args.no_fixed:
            combo = list(extra)
        else:
            combo = [fixed_idx] + list(extra)
        names = [solver_names[i] for i in extra]
        tasks.append((X, borda_matrix, problems, combo, names))

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): t[4] for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            names, ai_borda, oracle_borda, params = future.result()
            label = " + ".join(names)
            print(f"  [{i}/{len(combos)}] {fixed_label}{label}  AI={ai_borda:.1f}  oracle={oracle_borda:.1f}")
            results.append((names, ai_borda, oracle_borda, params))

    results.sort(key=lambda r: r[1], reverse=True)
    print(f"\n{'='*70}")
    print(f"Top 20 by AI Borda ({fixed_label}{args.n_extra} solvers):")
    print(f"{'='*70}")
    for rank, (names, ai, oracle, _) in enumerate(results[:20], 1):
        label = ", ".join(names)
        print(f"  #{rank:2d}  AI={ai:.1f}  oracle={oracle:.1f}  [{fixed_label}{label}]")


if __name__ == "__main__":
    main()
