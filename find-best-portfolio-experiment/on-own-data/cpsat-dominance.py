"""Count instances where cp-sat(8c) beats every other open-category solver."""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"

with open(COMBINED_CSV) as f:
    rows = list(csv.DictReader(f))

open_category = {(r["solver"], int(r["cores"])) for r in rows if r["open_category"] == "True"}
problem_types = load_problem_types(TYPES_CSV)
scores, configs, instances = borda_scores(rows, problem_types, opponents=open_category)

config_idx = {c: i for i, c in enumerate(configs)}
cpsat_idx = config_idx[("cp-sat", 8)]

oc_idxs = [i for i, c in enumerate(configs) if c in open_category and i != cpsat_idx]

cpsat_scores = scores[cpsat_idx]
others_max = scores[oc_idxs].max(axis=0)

strictly_best = np.sum(cpsat_scores > others_max)
tied_for_best = np.sum((cpsat_scores == others_max) & (cpsat_scores > 0))
best_or_tied = strictly_best + tied_for_best

print(f"Instances: {len(instances)}")
print(f"cp-sat(8c) strictly best: {strictly_best} ({100*strictly_best/len(instances):.1f}%)")
print(f"cp-sat(8c) tied for best: {tied_for_best} ({100*tied_for_best/len(instances):.1f}%)")
print(f"cp-sat(8c) best or tied:  {best_or_tied} ({100*best_or_tied/len(instances):.1f}%)")
