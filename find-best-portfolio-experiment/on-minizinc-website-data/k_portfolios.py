import json
from itertools import combinations

import numpy as np

from portfolio import open_solver_configs, open_competitors, build_score_matrix


def all_portfolios(data):
    configs = open_solver_configs(data)
    competitors = open_competitors(data)
    solvers = data["results"]["solvers"]
    M = build_score_matrix(data, competitors)

    single_core = [name for name, c in configs if c == 1]
    eight_core = [name for name, c in configs if c == 8]

    portfolios = []
    instance_scores = []

    for name in eight_core:
        per_instance = M[solvers.index(name)].sum(axis=0)
        portfolios.append([(name, 8)])
        instance_scores.append(per_instance)

    for combo in combinations(single_core, 8):
        idxs = [solvers.index(n) for n in combo]
        per_instance = M[idxs].max(axis=0).sum(axis=0)
        portfolios.append([(n, 1) for n in combo])
        instance_scores.append(per_instance)

    return portfolios, np.array(instance_scores)


def greedy_k_portfolios(portfolios, instance_scores, k):
    best_so_far = np.zeros(instance_scores.shape[1])
    selected = []
    for _ in range(k):
        gains = np.maximum(instance_scores - best_so_far, 0).sum(axis=1)
        idx = gains.argmax()
        selected.append(idx)
        best_so_far = np.maximum(best_so_far, instance_scores[idx])
    return selected, best_so_far.sum()


if __name__ == "__main__":
    with open("results.json") as f:
        data = json.load(f)

    portfolios, instance_scores = all_portfolios(data)
    print(f"{len(portfolios)} portfolios enumerated\n")

    for k in range(1, 10):
        selected, total = greedy_k_portfolios(portfolios, instance_scores, k)
        print(f"k={k}  score={total:.2f}")
        for i, idx in enumerate(selected):
            p = portfolios[idx]
            cores = sum(c for _, c in p)
            names = ", ".join(f"{n}({c}c)" for n, c in p)
            print(f"  portfolio {i+1}: [{names}]")
        print()
