"""Score a portfolio from a CSV schedule against all open-category solvers."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"

def load_portfolio(path):
    portfolio = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            solver, cores = line.rsplit(",", 1)
            portfolio.append((solver, int(cores)))
    return portfolio

portfolios = [load_portfolio(p) for p in sys.argv[1:]]

with open(COMBINED_CSV) as f:
    rows = list(csv.DictReader(f))

open_category = {(r["solver"], int(r["cores"])) for r in rows if r["open_category"] == "True"}
problem_types = load_problem_types(TYPES_CSV)
scores, configs, instances = borda_scores(rows, problem_types, opponents=open_category)

config_idx = {c: i for i, c in enumerate(configs)}

# Build wrong matrix
instance_idx = {k: i for i, k in enumerate(instances)}
wrong = np.zeros((len(configs), len(instances)), dtype=bool)
for r in rows:
    if r.get("wrong") == "True":
        ci = config_idx[(r["solver"], int(r["cores"]))]
        ii = instance_idx[(r["problem"], r["name"])]
        wrong[ci, ii] = True

# Global oracle: best possible score across all open-category solvers
oc_idxs = [i for i, c in enumerate(configs) if c in open_category]
global_oracle = scores[oc_idxs].max(axis=0)

# Per-portfolio scores
portfolio_score_vecs = []
for i, portfolio in enumerate(portfolios, 1):
    pidxs = [config_idx[c] for c in portfolio]
    pscores = scores[pidxs].max(axis=0)
    any_wrong = wrong[pidxs].any(axis=0)
    pscores[any_wrong] = 0
    portfolio_score_vecs.append(pscores)

    cores = sum(c for _, c in portfolio)
    names = ", ".join(f"{s}({c}c)" for s, c in portfolio)
    print(f"Portfolio {i} ({cores}c): {names}")
    print(f"  Score: {pscores.sum():.2f}")

# Oracle choosing best portfolio per instance
oracle_scores = np.maximum.reduce(portfolio_score_vecs)
matches_global = np.sum(oracle_scores >= global_oracle)
n = len(instances)
print(f"\nInstances: {n}")
print(f"Global oracle score:    {global_oracle.sum():.2f}")
print(f"Portfolio oracle score: {oracle_scores.sum():.2f}")
print(f"Matches global oracle:  {matches_global}/{n} ({100*matches_global/n:.1f}%)")
print(f"Falls short:            {n - matches_global}/{n} ({100*(n - matches_global)/n:.1f}%)")
