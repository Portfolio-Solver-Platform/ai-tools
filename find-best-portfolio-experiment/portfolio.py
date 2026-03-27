import json
from itertools import combinations

import numpy as np

from systems import solver_systems


def open_solver_configs(data):
    r = data["results"]
    systems = solver_systems(data)
    open_names = {s for s, o in zip(r["solvers"], r["open_solvers"]) if o}
    open_systems = {base for base, info in systems.items() if any(n in open_names for n in info["benchmarks"].values())}
    configs = []
    for base in open_systems:
        for cores, name in systems[base]["benchmarks"].items():
            configs.append((name, cores))
    return configs


def open_competitors(data):
    r = data["results"]
    return [s for s, o in zip(r["solvers"], r["open_solvers"]) if o]


def build_score_matrix(data, competitor_names):
    r = data["results"]
    comp_idxs = [r["solvers"].index(n) for n in competitor_names]
    return np.array(r["scores"])[:, comp_idxs, :]


def best_portfolios(data, top_n=100):
    configs = open_solver_configs(data)
    competitors = open_competitors(data)
    solvers = data["results"]["solvers"]
    M = build_score_matrix(data, competitors)

    single_core = [name for name, c in configs if c == 1]
    eight_core = [name for name, c in configs if c == 8]

    candidates = []

    for name in eight_core:
        s = M[solvers.index(name)].sum()
        candidates.append((s, [(name, 8)]))
        
    # for i in combinations(single_core)

    for combo in combinations(single_core, 8):
        idxs = [solvers.index(n) for n in combo]
        s = M[idxs].max(axis=0).sum()
        candidates.append((s, [(n, 1) for n in combo]))

    candidates.sort(reverse=True)
    return candidates[:top_n]


if __name__ == "__main__":
    with open("results.json") as f:
        data = json.load(f)

    for rank, (s, portfolio) in enumerate(best_portfolios(data), 1):
        cores = sum(c for _, c in portfolio)
        names = ", ".join(f"{n}({c}c)" for n, c in portfolio)
        print(f"#{rank:2d}  score={s:7.2f}  cores={cores}  {names}")
