import json
from itertools import combinations

import numpy as np

from systems import solver_systems


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


# def open_competitors(data):
#     r = data["results"]
#     return [s for s, o in zip(r["solvers"], r["open_solvers"]) if o]


def all_competitors_with_cores(data):
      r = data["results"]
      systems = solver_systems(data)
      configs = []
      for base in systems:
          for cores, name in systems[base]["benchmarks"].items():
              configs.append((name, cores))
      return configs

def build_score_matrix(data, competitor_names):
    r = data["results"]
    comp_idxs = [r["solvers"].index(n) for n in competitor_names]
    return np.array(r["scores"])[:, comp_idxs, :]


def best_portfolios(data, top_n=100000):
    competitors = all_competitors_with_cores(data)
    solvers = data["results"]["solvers"]
    # (solvers, opponents, instances) → (solvers, instances)
    per_instance = build_score_matrix(data, solvers).sum(axis=1)

    candidates = []
    for combo in combinations_summing_to(competitors, 8, lambda k: k[1]):
        idxs = [solvers.index(name) for name, _ in combo]
        score = per_instance[idxs].max(axis=0).sum()
        candidates.append((score, list(combo)))

    candidates.sort(reverse=True)
    return candidates[:top_n]


if __name__ == "__main__":
    with open("results.json") as f:
        data = json.load(f)

    for rank, (s, portfolio) in enumerate(best_portfolios(data), 1):
        cores = sum(c for _, c in portfolio)
        names = ", ".join(f"{n}({c}c)" for n, c in portfolio)
        print(f"#{rank:2d}  score={s:7.2f}  cores={cores}  {names}")
