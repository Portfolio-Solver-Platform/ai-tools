"""
Sweep k=1..6 and find the best combo by oracle+floor and oracle+mean.

- Oracle: max per instance (AI picks perfectly)
- Floor:  min per instance (AI picks worst)
- Mean:   mean per instance (AI picks randomly)

Usage:
    python best-k-sweep.py
"""
import csv
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
PORTFOLIO_CSV = ROOT / "benchmarks/portfolios/combined.csv"
OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"

CORES = 8
MAX_K = 5


def load_portfolio_rows(path):
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


def find_best(score_vecs, k, metrics, top_n=10):
    """Find the top_n combos of k for each metric. Returns {name: [(val, oracle, floor, mean, combo), ...]}."""
    import heapq
    n = score_vecs.shape[0]
    heaps = {name: [] for name in metrics}

    for combo in combinations(range(n), k):
        mat = score_vecs[list(combo)]  # (k, n_instances)
        oracle = mat.max(axis=0).sum()
        floor = mat.min(axis=0).sum()
        mean = mat.mean(axis=0).sum()

        for name, fn in metrics.items():
            val = fn(oracle, floor, mean)
            entry = (val, oracle, floor, mean, combo)
            if len(heaps[name]) < top_n:
                heapq.heappush(heaps[name], entry)
            elif val > heaps[name][0][0]:
                heapq.heapreplace(heaps[name], entry)

    return {name: sorted(h, reverse=True) for name, h in heaps.items()}


if __name__ == "__main__":
    portfolio_rows = load_portfolio_rows(PORTFOLIO_CSV)
    open_rows = load_open_rows(OPEN_CSV)

    all_rows = portfolio_rows + open_rows
    problem_types = load_problem_types(TYPES_CSV)

    open_configs = {(r["solver"], int(r["cores"])) for r in open_rows if r["open_category"] == "True"}
    scores, configs, instances = borda_scores(all_rows, problem_types, opponents=open_configs)
    config_idx = {c: i for i, c in enumerate(configs)}

    portfolio_configs = [(s, CORES) for s in sorted(set(r["solver"] for r in portfolio_rows))]
    solver_8c_configs = sorted(c for c in open_configs if c[1] == CORES)
    all_candidates = portfolio_configs + solver_8c_configs
    names = [c[0] for c in all_candidates]
    idxs = [config_idx[c] for c in all_candidates]
    score_vecs = scores[idxs]
    solver_set = {c[0] for c in solver_8c_configs}

    print(f"Candidates: {len(names)} ({len(portfolio_configs)} portfolios + {len(solver_8c_configs)} solvers)")
    print(f"Instances: {len(instances)}")

    metrics = {
        "oracle+floor":      lambda o, f, m: o + f,
        "oracle+mean":       lambda o, f, m: o + m,
        "oracle+floor+mean": lambda o, f, m: o + f + m,
    }

    all_results = {name: [] for name in metrics}  # metric -> [(val, oracle, floor, mean, labels, k)]

    for k in range(1, MAX_K + 1):
        total_combos = math.comb(len(names), k)
        print(f"\n{'='*60}")
        print(f"k={k}  ({total_combos} combinations)")

        per_k = find_best(score_vecs, k, metrics, top_n=10)

        for metric_name in metrics:
            val, oracle, floor, mean, combo = per_k[metric_name][0]
            picked = [names[i] for i in combo]
            labels = [f"{n} *" if n in solver_set else n for n in picked]

            print(f"\n  Best by {metric_name}:")
            print(f"    oracle={oracle:.1f}  floor={floor:.1f}  mean={mean:.1f}")
            for label in labels:
                print(f"      {label}")

            for val, oracle, floor, mean, combo in per_k[metric_name]:
                picked = [names[i] for i in combo]
                lbls = [f"{n} *" if n in solver_set else n for n in picked]
                all_results[metric_name].append((val, oracle, floor, mean, lbls, k))

    # Overall top across all k values
    for metric_name in metrics:
        all_results[metric_name].sort(key=lambda r: r[0], reverse=True)
        # Deduplicate (same combo can appear via different metrics)
        seen = set()
        deduped = []
        for r in all_results[metric_name]:
            key = tuple(sorted(r[4]))
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        top_n = 10 if metric_name == "oracle+floor+mean" else 3
        print(f"\n{'='*60}")
        print(f"Overall top {top_n} by {metric_name}:")
        for rank, (val, oracle, floor, mean, labels, k) in enumerate(deduped[:top_n], 1):
            names_str = ", ".join(labels)
            print(f"  #{rank}  k={k}  oracle={oracle:.1f}  floor={floor:.1f}  mean={mean:.1f}  [{names_str}]")
