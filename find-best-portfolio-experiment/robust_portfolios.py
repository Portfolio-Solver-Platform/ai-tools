import json
import sys
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
        portfolios.append([(name, 8)])
        instance_scores.append(M[solvers.index(name)].sum(axis=0))

    for combo in combinations(single_core, 8):
        idxs = [solvers.index(n) for n in combo]
        portfolios.append([(n, 1) for n in combo])
        instance_scores.append(M[idxs].max(axis=0).sum(axis=0))

    return portfolios, np.array(instance_scores)


def robust_score(selected_scores, accuracy):
    return (accuracy * selected_scores.max(axis=0) + (1 - accuracy) * selected_scores.mean(axis=0)).sum()


def greedy_k_portfolios(instance_scores, k, accuracy):
    n_portfolios = instance_scores.shape[0]
    best_idx = instance_scores.sum(axis=1).argmax()
    selected = [best_idx]

    for _ in range(k - 1):
        best_gain = -np.inf
        best_next = -1
        for p in range(n_portfolios):
            candidate = instance_scores[selected + [p]]
            obj = robust_score(candidate, accuracy)
            if obj > best_gain:
                best_gain = obj
                best_next = p
        selected.append(best_next)

    return selected, robust_score(instance_scores[selected], accuracy)


if __name__ == "__main__":
    accuracy = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90

    with open("results.json") as f:
        data = json.load(f)

    portfolios, instance_scores = all_portfolios(data)
    print(f"{len(portfolios)} portfolios, accuracy={accuracy:.0%}\n")

    for k in range(1, 7):
        selected, total = greedy_k_portfolios(instance_scores, k, accuracy)
        print(f"k={k:2d}  score={total:.2f}")
        for i, idx in enumerate(selected):
            p = portfolios[idx]
            names = ", ".join(f"{n}({c}c)" for n, c in p)
            print(f"       portfolio {i+1}: [{names}]")
        print()
