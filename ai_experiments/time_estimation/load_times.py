"""Build a (N, n_portfolios) time matrix aligned with the npz meta,
sourcing time_ms from benchmarks/portfolios/final-portfolios/combined.csv.
Missing portfolio rows for an instance are filled with `timeout_ms`.
"""
import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMBINED_CSV = REPO_ROOT / "benchmarks" / "portfolios" / "final-portfolios" / "combined.csv"
PROBLEM_TYPES_CSV = REPO_ROOT / "benchmarks" / "open-category-benchmarks" / "problem_types.csv"


def build_Y_times(meta, portfolios=('cpsat8', 'k1-8c-8s-v1', 'ek1-8c-8s-v2'), timeout_ms=1_200_000):
    """Look up time_ms for each (instance, portfolio) and return an aligned matrix.

    meta:        structured array from the npz with fields year/problem/model/name
    portfolios:  tuple of solver names (column order in returned matrix)
    timeout_ms:  fill value used when a portfolio has no row for an instance, or
                 when time_ms is empty in combined.csv
    """
    times_by_key = {}
    with open(COMBINED_CSV, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['year'], r['problem'], r['model'], r['name'])
            t = float(r['time_ms']) if r['time_ms'] != '' else timeout_ms
            times_by_key.setdefault(key, {})[r['solver']] = t

    n = len(meta)
    Y_times = np.full((n, len(portfolios)), timeout_ms, dtype=float)
    n_missing = 0
    for i in range(n):
        key = (str(meta['year'][i]), str(meta['problem'][i]), str(meta['model'][i]), str(meta['name'][i]))
        per_solver = times_by_key.get(key, {})
        for k, p in enumerate(portfolios):
            if p in per_solver:
                Y_times[i, k] = per_solver[p]
            else:
                n_missing += 1
    return Y_times, n_missing


def build_Y_times_with_tiebreak(meta, portfolios=('cpsat8', 'k1-8c-8s-v1', 'ek1-8c-8s-v2'),
                                  timeout_ms=1_200_000, tiebreak_penalty_ms=1):
    """Like build_Y_times, but on rows where all selected solvers timed out, apply a small
    time penalty to solvers whose final objective is worse than the best found among them.
    Only applied for MIN/MAX problems where at least one solver recorded a feasible objective.
    SAT problems and rows with no recorded objectives stay tied.
    """
    times_by_key = {}
    objs_by_key = {}
    with open(COMBINED_CSV, newline='') as f:
        for r in csv.DictReader(f):
            key = (r['year'], r['problem'], r['model'], r['name'])
            t = float(r['time_ms']) if r['time_ms'] != '' else timeout_ms
            times_by_key.setdefault(key, {})[r['solver']] = t
            objs_by_key.setdefault(key, {})[r['solver']] = r['objective']

    problem_types = {}
    with open(PROBLEM_TYPES_CSV, newline='') as f:
        for r in csv.DictReader(f):
            problem_types[(r['problem'], r['model'])] = r['type']

    n = len(meta)
    Y_times = np.full((n, len(portfolios)), timeout_ms, dtype=float)
    n_missing = 0
    n_tiebroken = 0
    n_genuine_tie = 0

    for i in range(n):
        prob = str(meta['problem'][i])
        model = str(meta['model'][i])
        key = (str(meta['year'][i]), prob, model, str(meta['name'][i]))
        per_solver = times_by_key.get(key, {})
        for k, p in enumerate(portfolios):
            if p in per_solver:
                Y_times[i, k] = per_solver[p]
            else:
                n_missing += 1

        all_timed_out = all(Y_times[i, k] >= timeout_ms for k in range(len(portfolios)))
        if not all_timed_out:
            continue

        kind = problem_types.get((prob, model))
        if kind not in ('MIN', 'MAX'):
            n_genuine_tie += 1
            continue

        obj_per_solver = objs_by_key.get(key, {})
        solver_objs = []
        for k, p in enumerate(portfolios):
            obj_str = obj_per_solver.get(p, '')
            if obj_str == '':
                solver_objs.append((k, None))
            else:
                solver_objs.append((k, float(obj_str)))

        feasible = [(k, o) for k, o in solver_objs if o is not None]
        if not feasible:
            n_genuine_tie += 1
            continue

        if kind == 'MIN':
            best_obj = min(o for _, o in feasible)
            is_worse = lambda o: o is None or o > best_obj
        else:
            best_obj = max(o for _, o in feasible)
            is_worse = lambda o: o is None or o < best_obj

        for k, o in solver_objs:
            if is_worse(o):
                Y_times[i, k] += tiebreak_penalty_ms
        n_tiebroken += 1

    return Y_times, n_missing, n_tiebroken, n_genuine_tie
