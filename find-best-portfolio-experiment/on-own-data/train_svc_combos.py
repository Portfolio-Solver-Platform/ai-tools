"""
For each portfolio combo (cp-sat(8c) + one portfolio schedule), train an SVC
to pick the best schedule per instance, evaluated by Borda score.

Uses GroupKFold (k=5) by problem so test problems are never seen during training.
Optuna tunes C/gamma per combo.

Usage:
    python train_svc_combos.py
"""
import csv
import pickle
import sys
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
PORTFOLIO_CSV = ROOT / "benchmarks/portfolios/combined.csv"
OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
DATA_DIR = ROOT / "data"

CORES = 8
N_FOLDS = 5
N_TRIALS = 50
FIXED = ("cp-sat", CORES)  # always included


def load_features():
    """Load and merge feature dicts from both pickle files."""
    features = {}
    for name in ["23_instances_features.pkl", "24_25_instances_features.pkl"]:
        with open(DATA_DIR / name, "rb") as f:
            features.update(pickle.load(f))
    return features


def instance_key(row):
    m, n = row["model"], row["name"]
    return f"{m}_{n}" if m != n else f"{m}_"


def build_data(portfolio_csv, open_csv, types_csv):
    """Build feature matrix X, Borda score matrix, instance metadata."""
    # Load all rows and compute Borda scores
    with open(portfolio_csv) as f:
        portfolio_rows = list(csv.DictReader(f))
    with open(open_csv) as f:
        open_rows = list(csv.DictReader(f))

    # Adapt portfolio rows for borda_scores
    adapted = []
    for r in portfolio_rows:
        adapted.append({
            "solver": r["schedule"], "cores": CORES,
            "problem": r["problem"], "name": r["name"], "model": r["model"],
            "status": r["status"], "time_ms": r["time_ms"],
            "objective": r["objective"], "wrong": r["wrong"],
        })

    all_rows = adapted + open_rows
    problem_types = load_problem_types(types_csv)
    open_configs = {(r["solver"], int(r["cores"])) for r in open_rows if r["open_category"] == "True"}

    scores, configs, instances = borda_scores(all_rows, problem_types, opponents=open_configs)
    config_idx = {c: i for i, c in enumerate(configs)}

    # Load features
    features = load_features()

    # Build aligned arrays: only instances that have features
    schedules = sorted(set(r["schedule"] for r in portfolio_rows))
    schedule_configs = [(s, CORES) for s in schedules]

    # Instance ordering: (problem, name) from borda output
    instance_keys_list = []  # feature keys
    instance_problems = []   # problem name (for grouping)
    instance_idxs = []       # index into borda scores array
    for i, (problem, name) in enumerate(instances):
        # Find model for this instance
        model = None
        for r in portfolio_rows:
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

    # Borda score matrix: (n_schedules, n_instances) for our instances
    all_schedule_configs = [FIXED] + schedule_configs
    all_schedule_names = [FIXED[0]] + schedules
    borda_matrix = np.zeros((len(all_schedule_configs), len(inst_idxs)))
    for si, cfg in enumerate(all_schedule_configs):
        ci = config_idx[cfg]
        borda_matrix[si] = scores[ci, inst_idxs]

    return X, borda_matrix, problems, all_schedule_names


def evaluate_combo(X, borda_matrix, problems, schedule_indices):
    """Run Optuna + GroupKFold for a specific combo of schedule indices.
    Returns (best_borda, oracle_borda, optuna_study)."""
    combo_borda = borda_matrix[schedule_indices]  # (k, n_instances)
    labels = np.argmax(combo_borda, axis=0)       # best schedule per instance
    oracle_borda = combo_borda.max(axis=0).sum()

    groups = problems
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(X, labels, groups))

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
            # Sum Borda of predicted schedule for each test instance
            total_borda += sum(combo_borda[preds[i], test_idx[i]] for i in range(len(test_idx)))

        return total_borda

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    return study.best_value, oracle_borda, study


def main():
    print("Loading data...")
    X, borda_matrix, problems, schedule_names = build_data(PORTFOLIO_CSV, OPEN_CSV, TYPES_CSV)
    print(f"Instances: {X.shape[0]}, Features: {X.shape[1]}, Schedules: {len(schedule_names)}")

    cpsat_idx = schedule_names.index(FIXED[0])
    cpsat_solo_borda = borda_matrix[cpsat_idx].sum()
    print(f"cp-sat(8c) solo Borda: {cpsat_solo_borda:.1f}")

    # Evaluate each portfolio paired with cp-sat
    results = []
    portfolio_indices = [i for i, n in enumerate(schedule_names) if n != FIXED[0]]

    for pi in portfolio_indices:
        name = schedule_names[pi]
        combo = [cpsat_idx, pi]
        solo_borda = borda_matrix[pi].sum()
        print(f"\n--- cp-sat + {name} (solo={solo_borda:.1f}) ---")

        ai_borda, oracle_borda, study = evaluate_combo(X, borda_matrix, problems, combo)
        print(f"  AI={ai_borda:.1f}  oracle={oracle_borda:.1f}  C={study.best_params['C']:.3f}  gamma={study.best_params['gamma']:.4f}")
        results.append((name, ai_borda, oracle_borda, solo_borda, study.best_params))

    # Summary
    results.sort(key=lambda r: r[1], reverse=True)
    print(f"\n{'='*70}")
    print(f"{'Schedule':<35s} {'AI':>8s} {'Oracle':>8s} {'Solo':>8s} {'cp-sat':>8s}")
    print(f"{'='*70}")
    for name, ai, oracle, solo, _ in results:
        print(f"{name:<35s} {ai:8.1f} {oracle:8.1f} {solo:8.1f} {cpsat_solo_borda:8.1f}")


if __name__ == "__main__":
    main()
