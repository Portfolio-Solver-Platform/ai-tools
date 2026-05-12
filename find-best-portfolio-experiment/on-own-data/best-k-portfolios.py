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
    # "org.chuffed.chuffed",
    # "org.gecode.gecode",
    # "solutions.huub",
}

MAX_SOLVERS_PER_PORTFOLIO = 8
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
    n = len(configs)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i: at least as good everywhere and uses no more cores
            if configs[j][1] <= configs[i][1] and np.all(scores[j] >= scores[i]):
                dominated[i] = True
                break
    pruned = [c for c, d in zip(configs, dominated) if not d]
    removed = {c[0] for c, d in zip(configs, dominated) if d} - {c[0] for c in pruned}
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
    """Find the top-N k-combinations of portfolios by oracle score.

    Vectorised over the innermost index: the k-1 outer indices iterate in
    Python, but for each prefix the score against every remaining candidate
    is computed in one numpy max+sum, then a vectorised mask filters to
    candidates that beat the current heap floor before pushing.
    """
    score_vecs = np.array([p[0] for p in portfolios], dtype=np.float64)
    n, m = score_vecs.shape
    base_vec = (baseline if baseline is not None
                else np.full(m, -np.inf, dtype=np.float64))
    total = math.comb(n, k)

    heap = []  # min-heap of (score, combo_tuple); never grows past heap_size
    progress = {"count": 0}
    leaves_per_dot = max(1, total // 1000)

    def push_batch(scores_arr, prefix_idxs, start):
        """Push leaves indices [start, start+len(scores_arr)) into the heap."""
        leaves = len(scores_arr)
        progress["count"] += leaves
        floor = heap[0][0] if len(heap) >= heap_size else -np.inf
        winners = np.flatnonzero(scores_arr > floor)
        for off in winners:
            entry = (float(scores_arr[off]), prefix_idxs + (start + int(off),))
            if len(heap) < heap_size:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
        if progress["count"] // leaves_per_dot != (progress["count"] - leaves) // leaves_per_dot:
            print(f"\r  {progress['count']}/{total} "
                  f"({100*progress['count']/total:.1f}%)",
                  end="", flush=True)

    def recurse(prefix_idxs, prefix_max, start):
        depth = len(prefix_idxs)
        remaining = k - depth
        if remaining == 0:
            # Leaf with no remaining selections — only happens if k == 0.
            return
        if remaining == 1:
            if start >= n:
                return
            # max(prefix_max, score_vecs[i]) for every i >= start
            inner = np.maximum(prefix_max, score_vecs[start:])
            scores = inner.sum(axis=1)
            push_batch(scores, prefix_idxs, start)
        else:
            # Iterate the next index in Python; recurse into the rest.
            limit = n - remaining + 1
            for i in range(start, limit):
                new_max = np.maximum(prefix_max, score_vecs[i])
                recurse(prefix_idxs + (i,), new_max, i + 1)

    recurse((), base_vec, 0)
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
