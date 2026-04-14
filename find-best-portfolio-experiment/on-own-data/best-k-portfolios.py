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
COMBINED_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"

FIXED_PORTFOLIOS = [
    [("cp-sat", 8)],
]

EXCLUDED_SOLVERS = {
    # "org.minizinc.mip.coin-bc",
    # "org.minizinc.mip.highs",
    # "izplus",
    # "org.choco.choco",
}

MAX_SOLVERS_PER_PORTFOLIO = 4
MAX_CORES = 8


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


def prune_configs(scores, configs):
    best_per_instance = scores.argmax(axis=0)
    best_solvers = {configs[i][0] for i in best_per_instance}
    removed = {c[0] for c in configs} - best_solvers
    pruned = [c for c in configs if c[0] in best_solvers]
    print(f"Configs: {len(configs)} → {len(pruned)} (removed solvers: {removed or 'none'})")
    return pruned


def generate_portfolios(configs, config_idx, scores, wrong=None):
    portfolios = []

    def portfolio_score(idxs):
        score_vec = scores[idxs].max(axis=0)
        if wrong is not None:
            any_wrong = wrong[idxs].any(axis=0)
            score_vec[any_wrong] = 0
        return score_vec

    # Single 8-core solver = a complete portfolio on its own
    for c in configs:
        if c[1] == 8:
            portfolios.append((portfolio_score([config_idx[c]]), [c]))

    # cp-sat(1c) + others summing to 7 (no duplicate solvers per portfolio)
    others = [c for c in configs if c[0] != "cp-sat"]
    for combo in combinations_summing_to(others, MAX_CORES-1, key=lambda c: c[1]):
        solvers = [s for s, _ in combo]
        if len(solvers) != len(set(solvers)):
            continue
        full = [("cp-sat", 1)] + list(combo)
        if len(full) > MAX_SOLVERS_PER_PORTFOLIO:
            continue
        idxs = [config_idx[c] for c in full]
        portfolios.append((portfolio_score(idxs), full))

    print(f"Generated {len(portfolios)} robust portfolios")
    return portfolios


def prune_portfolios(portfolios, baseline=None):
    if baseline is not None:
        score_vecs = np.maximum(np.array([p[0] for p in portfolios]), baseline)
    else:
        score_vecs = np.array([p[0] for p in portfolios])
    order = np.argsort(-score_vecs.sum(axis=1))  # best first

    remaining = list(order)
    surviving = []
    while remaining:
        best = remaining[0]
        surviving.append(best)
        rest = remaining[1:]
        if not rest:
            break
        dominated = np.all(score_vecs[best] >= score_vecs[rest], axis=1)
        remaining = [r for r, d in zip(rest, dominated) if not d]
        if len(surviving) % 100 == 0:
            print(f"\r  Dominance check: {len(surviving)} surviving, {len(remaining)} remaining", end="", flush=True)
    print()

    pruned = [portfolios[i] for i in surviving]
    print(f"Dominance pruning: {len(portfolios)} → {len(pruned)}")
    return pruned


def best_k_portfolios(portfolios, k=1, top_n=30, heap_size=500, baseline=None):
    score_vecs = np.array([p[0] for p in portfolios])
    n = len(portfolios)
    heap = []
    combos = combinations(range(n), k)

    def score_combo(combo):
        s = score_vecs[list(combo)].max(axis=0)
        if baseline is not None:
            s = np.maximum(s, baseline)
        return s.sum()

    for combo in itertools.islice(combos, heap_size):
        heapq.heappush(heap, (score_combo(combo), combo))

    total = math.comb(n, k)
    for count, combo in enumerate(combos, heap_size + 1):
        score = score_combo(combo)
        if score > heap[0][0]:
            heapq.heapreplace(heap, (score, combo))
        if count % 100_000 == 0:
            print(f"\r  {count}/{total} ({100*count/total:.1f}%)", end="", flush=True)
    print()

    # Tiebreaker: prefer combos where the weakest portfolio is strongest
    ranked = []
    for score, combo in heap:
        min_individual = min(score_vecs[i].sum() for i in combo)
        ranked.append((score, min_individual, combo))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)

    return [(s, robustness, [portfolios[i][1] for i in idxs])
            for s, robustness, idxs in ranked[:top_n]]


if __name__ == "__main__":
    with open(COMBINED_CSV) as f:
        rows = list(csv.DictReader(f))

    open_category = {(r["solver"], int(r["cores"])) for r in rows if r["open_category"] == "True"}

    problem_types = load_problem_types(TYPES_CSV)
    scores, configs, instances = borda_scores(rows, problem_types, opponents=open_category)
    config_idx = {c: i for i, c in enumerate(configs)}
    instance_idx = {k: i for i, k in enumerate(instances)}

    # Build wrong matrix: True where solver gave provably incorrect answer
    wrong = np.zeros((len(configs), len(instances)), dtype=bool)
    for r in rows:
        if r.get("wrong") == "True":
            ci = config_idx[(r["solver"], int(r["cores"]))]
            ii = instance_idx[(r["problem"], r["name"])]
            wrong[ci, ii] = True

    configs = prune_configs(scores, configs)
    configs = [c for c in configs if c[0] not in EXCLUDED_SOLVERS]

    portfolios = generate_portfolios(configs, config_idx, scores, wrong=wrong)

    # Extract fixed portfolios and compute baseline
    baseline = None
    for fixed in FIXED_PORTFOLIOS:
        fixed_vec = scores[[config_idx[c] for c in fixed]].max(axis=0)
        baseline = fixed_vec if baseline is None else np.maximum(baseline, fixed_vec)
        portfolios = [(v, p) for v, p in portfolios if p != fixed]

    portfolios = prune_portfolios(portfolios, baseline=baseline)

    k_extra = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    results = best_k_portfolios(portfolios, k=k_extra, baseline=baseline)

    for rank, (s, robustness, k_portfolio) in enumerate(results, 1):
        print(f"\n#{rank:2d}  oracle_score={s:.2f}  robustness={robustness:.2f}")
        for i, fixed in enumerate(FIXED_PORTFOLIOS):
            names = ", ".join(f"{solver}({c}c)" for solver, c in fixed)
            print(f"    Portfolio {i+1} (fixed): {names}")
        for i, portfolio in enumerate(k_portfolio, len(FIXED_PORTFOLIOS) + 1):
            cores = sum(c for _, c in portfolio)
            names = ", ".join(f"{solver}({c}c)" for solver, c in portfolio)
            print(f"    Portfolio {i} ({cores}c): {names}")
