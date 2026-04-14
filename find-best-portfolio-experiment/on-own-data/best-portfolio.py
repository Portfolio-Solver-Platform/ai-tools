import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"


def combinations_summing_to(items, target, key=lambda x: x):
    def backtrack(start, remaining, current):
        if remaining == 0:
            yield tuple(current)
            return
        for i in range(start, len(items)):
            val = key(items[i])
            if val > remaining:
                continue
            current.append(items[i])
            yield from backtrack(i + 1, remaining - val, current)
            current.pop()

    yield from backtrack(0, target, [])


def best_portfolios(scores, configs, config_idx, top_n=100):
    candidates = []
    for combo in combinations_summing_to(configs, 8, lambda k: k[1]):
        idxs = [config_idx[c] for c in combo]
        score = scores[idxs].max(axis=0).sum()
        candidates.append((score, list(combo)))

    candidates.sort(reverse=True)
    return candidates[:top_n]


if __name__ == "__main__":
    with open(COMBINED_CSV) as f:
        rows = list(csv.DictReader(f))

    problem_types = load_problem_types(TYPES_CSV)
    scores, configs, instances = borda_scores(rows, problem_types)
    config_idx = {c: i for i, c in enumerate(configs)}

    for rank, (s, portfolio) in enumerate(best_portfolios(scores, configs, config_idx), 1):
        cores = sum(c for _, c in portfolio)
        names = ", ".join(f"{solver}({c}c)" for solver, c in portfolio)
        print(f"#{rank:2d}  score={s:7.2f}  cores={cores}  {names}")
