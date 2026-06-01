"""Build training data where Y is each portfolio's per-instance Borda computed against
all open-category solvers (a richer signal than 2- or 3-solver pairwise Borda).

For each instance from years 2023-2025:
  - Load all open-category solvers' results (open_category=True rows in open-category combined.csv).
  - Load cp-sat (= cpsat8 portfolio), k1-8c-8s-v1, ek1-8c-8s-v2.
  - Compute Borda for each portfolio by tournament against every other solver row in that instance.
  - X comes from the per-year mznc{year}_features.pkl.

Returns X_train, Y_train_wide ((N, 3) Borda for [cpsat, k1, ek1]), meta_train.
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "scoring"))
from borda import pairwise_score, load_problem_types

DATA_DIR     = REPO_ROOT / "data"
OPEN_CSV     = REPO_ROOT / "benchmarks" / "open-category-benchmarks" / "combined.csv"
ALL_CSV      = REPO_ROOT / "benchmarks" / "portfolios" / "all" / "combined.csv"
ELIGIBLE_CSV = REPO_ROOT / "benchmarks" / "portfolios" / "eligible" / "combined.csv"
TYPES_CSV    = REPO_ROOT / "benchmarks" / "open-category-benchmarks" / "problem_types.csv"

TRAIN_YEARS = ('2024', '2025')
PORTFOLIO_LABELS = ['cp-sat', 'k1-8c-8s-v1', 'ek1-8c-8s-v2']  # column order in Y_train_wide


def make_key(problem, model, name):
    # Same convention as utils/raw_data_to_training_data/build_training_data.py.
    return f"{problem}_{model}_" if model == name else f"{problem}_{model}_{name}"


def load_wide_borda():
    problem_types = load_problem_types(TYPES_CSV)

    # instance_key -> list of solver rows
    instances = {}

    with open(OPEN_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r['year'] not in TRAIN_YEARS or r['open_category'] != 'True':
                continue
            key = (r['year'], r['problem'], r['model'], r['name'])
            instances.setdefault(key, []).append(r)

    with open(ALL_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r['year'] not in TRAIN_YEARS or r['schedule'] != 'k1-8c-8s-v1':
                continue
            r['solver'] = 'k1-8c-8s-v1'
            key = (r['year'], r['problem'], r['model'], r['name'])
            instances.setdefault(key, []).append(r)

    with open(ELIGIBLE_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r['year'] not in TRAIN_YEARS or r['schedule'] != 'ek1-8c-8s-v2':
                continue
            r['solver'] = 'ek1-8c-8s-v2'
            key = (r['year'], r['problem'], r['model'], r['name'])
            instances.setdefault(key, []).append(r)

    feature_cache = {}
    for year in TRAIN_YEARS:
        pkl = DATA_DIR / f"mznc{year}_features.pkl"
        if pkl.exists():
            with open(pkl, 'rb') as f:
                feature_cache[year] = pickle.load(f)

    rows_X, rows_Y, rows_meta = [], [], []
    skipped = {'no_portfolio': 0, 'no_type': 0, 'no_features': 0}

    for inst_key, solvers in instances.items():
        year, problem, model, name = inst_key

        present = {s['solver'] for s in solvers}
        if not all(p in present for p in PORTFOLIO_LABELS):
            skipped['no_portfolio'] += 1
            continue

        kind = problem_types.get((problem, model))
        if kind is None:
            skipped['no_type'] += 1
            continue

        if year not in feature_cache:
            skipped['no_features'] += 1
            continue
        feat = feature_cache[year].get(make_key(problem, model, name))
        if feat is None:
            skipped['no_features'] += 1
            continue

        # Tournament: each portfolio's Borda = sum of pairwise wins vs every other solver row
        bordas = []
        for p_label in PORTFOLIO_LABELS:
            p_row = next(s for s in solvers if s['solver'] == p_label)
            score = 0.0
            for s_row in solvers:
                if s_row is p_row:
                    continue
                score += pairwise_score(p_row, s_row, kind)
            bordas.append(score)

        rows_X.append(np.asarray(feat).ravel())
        rows_Y.append(bordas)
        rows_meta.append(inst_key)

    print(f"wide-Borda training set built from years {TRAIN_YEARS}:")
    print(f"  total instances kept: {len(rows_X)}")
    for k, v in skipped.items():
        print(f"  skipped ({k}): {v}")

    return np.array(rows_X), np.array(rows_Y), rows_meta
