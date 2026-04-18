"""
Find the best combination of k portfolio schedules by Borda score.

Unlike best-k-portfolios.py which generates hypothetical portfolios from
individual solver configs, this uses actual measured portfolio results.

Usage:
    python best-k-schedules.py <all|eligible> [k]
"""
import argparse
import csv
import heapq
import itertools
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"

# Schedules to always include (leave empty to find best k standalone)
FIXED_SCHEDULES = []

CORES = 8  # all portfolios run on 8 cores


def load_portfolio_rows(path):
    """Load portfolio CSV and adapt columns for borda_scores."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "solver": r["schedule"],
                "cores": CORES,
                "problem": r["problem"],
                "name": r["name"],
                "model": r["model"],
                "status": r["status"],
                "time_ms": r["time_ms"],
                "objective": r["objective"],
                "wrong": r["wrong"],
            })
    return rows


def load_open_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def best_k_schedules(score_vecs, schedules, k=1, top_n=30, baseline=None):
    n = len(schedules)
    heap = []

    def score_combo(combo):
        s = score_vecs[list(combo)].max(axis=0)
        if baseline is not None:
            s = np.maximum(s, baseline)
        return s.sum()

    heap_size = min(max(top_n, 500), math.comb(n, k))
    combos = combinations(range(n), k)

    for combo in itertools.islice(combos, heap_size):
        heapq.heappush(heap, (score_combo(combo), combo))

    total = math.comb(n, k)
    for count, combo in enumerate(combos, heap_size + 1):
        score = score_combo(combo)
        if score > heap[0][0]:
            heapq.heapreplace(heap, (score, combo))
        if count % 100_000 == 0:
            print(f"\r  {count}/{total} ({100*count/total:.1f}%)", end="", flush=True)
    if total > heap_size:
        print()

    ranked = []
    for score, combo in heap:
        min_individual = min(score_vecs[i].sum() for i in combo)
        ranked.append((score, min_individual, combo))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [(s, robustness, [schedules[i] for i in idxs])
            for s, robustness, idxs in ranked[:top_n]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["all", "eligible"],
                        help="Which portfolio dataset to use")
    parser.add_argument("k", nargs="?", type=int, default=1,
                        help="Number of extra schedules (default: 1)")
    args = parser.parse_args()

    portfolio_csv = ROOT / "benchmarks" / "portfolios" / args.dataset / "combined.csv"
    portfolio_rows = load_portfolio_rows(portfolio_csv)
    open_rows = load_open_rows(OPEN_CSV)

    # Combine: portfolios compete against open-category solvers
    all_rows = portfolio_rows + open_rows
    problem_types = load_problem_types(TYPES_CSV)

    open_configs = {(r["solver"], int(r["cores"])) for r in open_rows if r["open_category"] == "True"}
    scores, configs, instances = borda_scores(all_rows, problem_types, opponents=open_configs)

    config_idx = {c: i for i, c in enumerate(configs)}

    # Portfolio schedules + cp-sat 8-core solver
    portfolio_configs = [(s, CORES) for s in sorted(set(r["solver"] for r in portfolio_rows))]
    solver_8c_configs = sorted(c for c in open_configs if c[1] == CORES and c[0] == "cp-sat")
    all_candidates = portfolio_configs + solver_8c_configs
    schedule_names = [c[0] for c in all_candidates]
    schedule_idxs = [config_idx[c] for c in all_candidates]
    schedule_scores = scores[schedule_idxs]
    n_portfolios = len(portfolio_configs)

    print(f"Dataset: {args.dataset}")
    print(f"Portfolios: {n_portfolios}, Solvers(8c): {len(solver_8c_configs)}, Instances: {len(instances)}")
    print(f"Open-category opponents: {len(open_configs)}")

    # Individual scores
    solver_set = {c[0] for c in solver_8c_configs}
    totals = [(schedule_scores[i].sum(), schedule_names[i]) for i in range(len(schedule_names))]
    totals.sort(reverse=True)
    print(f"\nIndividual scores:")
    for score, name in totals:
        marker = " *" if name in solver_set else ""
        print(f"  {name:40s} {score:.1f}{marker}")

    # Handle fixed schedules
    baseline = None
    remaining_idxs = list(range(len(schedule_names)))
    for fixed in FIXED_SCHEDULES:
        idx = schedule_names.index(fixed)
        fixed_vec = schedule_scores[idx]
        baseline = fixed_vec if baseline is None else np.maximum(baseline, fixed_vec)
        remaining_idxs = [i for i in remaining_idxs if i != idx]

    remaining_scores = schedule_scores[remaining_idxs]
    remaining_names = [schedule_names[i] for i in remaining_idxs]

    results = best_k_schedules(remaining_scores, remaining_names, k=args.k, baseline=baseline)

    for rank, (s, robustness, picked) in enumerate(results, 1):
        print(f"\n#{rank:2d}  score={s:.1f}  min_individual={robustness:.1f}")
        for fixed in FIXED_SCHEDULES:
            print(f"    (fixed) {fixed}")
        for name in picked:
            marker = " *" if name in solver_set else ""
            print(f"    {name}{marker}")

    # Top 3 by combined score + min_individual (rewards both quality and robustness)
    by_combined = sorted(results, key=lambda x: x[0] + x[1], reverse=True)[:3]
    print(f"\n{'='*50}")
    print(f"Top 3 by score + min_individual:")
    for rank, (s, robustness, picked) in enumerate(by_combined, 1):
        names = ", ".join(f"{n} *" if n in solver_set else n for n in picked)
        print(f"  #{rank}  score={s:.1f}  min={robustness:.1f}  combined={s+robustness:.1f}  [{names}]")
