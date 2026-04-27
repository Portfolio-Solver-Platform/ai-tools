"""Oracle score: best possible Borda score using all open-category solvers."""
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

# Zero out wrong answers
config_idx = {c: i for i, c in enumerate(configs)}
instance_idx = {k: i for i, k in enumerate(instances)}
for r in rows:
    if r.get("wrong") == "True":
        ci = config_idx[(r["solver"], int(r["cores"]))]
        ii = instance_idx[(r["problem"], r["name"])]
        scores[ci, ii] = 0

exclude = set()
include = set()
for arg in sys.argv[1:]:
    if arg.startswith("+"):
        solver, cores = arg[1:].rsplit(",", 1)
        include.add((solver, int(cores)))
    else:
        solver, cores = arg.rsplit(",", 1)
        exclude.add((solver, int(cores)))

oc_idxs = [i for i, c in enumerate(configs) if c in open_category and c not in exclude]
extra_idxs = [i for i, c in enumerate(configs) if c in include and c not in open_category]
all_idxs = oc_idxs + extra_idxs
oracle = scores[all_idxs].max(axis=0)

mods = []
if exclude: mods.append(f"excluded: {exclude}")
if include: mods.append(f"added: {include}")
print(f"Solvers: {len(all_idxs)}" + (f" ({', '.join(mods)})" if mods else ""))
print(f"Instances: {len(instances)}")
print(f"Oracle score: {oracle.sum():.2f}")
