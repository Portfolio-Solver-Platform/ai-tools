"""
MiniZinc challenge Borda count scoring.

Computes pairwise scores between all (solver, cores) configs on each instance.
The caller is responsible for filtering the result rows to only include the
configs that should compete — this module always scores all against all.

Usage:
    from utils.borda import borda_scores

    scores, configs, instances = borda_scores(rows, problem_types)
    # scores: np.ndarray of shape (n_configs, n_instances)
    # configs: list of (solver, cores) tuples
    # instances: list of (problem, name) tuples
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SOLVED = {"Optimal", "Satisfied", "Unsat"}
COMPLETE = {"Optimal", "Unsat"}


def load_problem_types(path):
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row["problem"], row["model"])] = row["type"]
    return types


def _split_by_time(time_a, time_b):
    total = time_a + time_b
    if total == 0:
        return 0.5, 0.5
    return time_b / total, time_a / total


def _compare(status_a, time_a, obj_a, status_b, time_b, obj_b, kind):
    a_solved = status_a in SOLVED
    b_solved = status_b in SOLVED

    if not a_solved and not b_solved:
        return 0.0, 0.0
    if a_solved and not b_solved:
        return 1.0, 0.0
    if not a_solved and b_solved:
        return 0.0, 1.0

    if kind == "SAT":
        return _split_by_time(time_a, time_b)

    if obj_a is not None and obj_b is not None:
        better_a = (kind == "MIN" and obj_a < obj_b) or (kind == "MAX" and obj_a > obj_b)
        better_b = (kind == "MIN" and obj_b < obj_a) or (kind == "MAX" and obj_b > obj_a)
        if better_a:
            return 1.0, 0.0
        if better_b:
            return 0.0, 1.0

    a_complete = status_a in COMPLETE
    b_complete = status_b in COMPLETE
    if a_complete and not b_complete:
        return 1.0, 0.0
    if not a_complete and b_complete:
        return 0.0, 1.0

    return _split_by_time(time_a, time_b)


def _parse_obj(s):
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def borda_scores(rows, problem_types, opponents=None):
    """
    Compute Borda scores from result rows.

    Args:
        rows: list of dicts with keys: solver, cores, problem, name, model,
              status, time_ms, objective, wrong
        problem_types: dict mapping (problem, model) -> "MIN"/"MAX"/"SAT"
        opponents: optional set of (solver, cores) tuples. If given, only
                   pairwise comparisons against these configs count.

    Returns:
        (scores, configs, instances) where:
            scores: np.ndarray of shape (n_configs, n_instances)
            configs: list of (solver, cores) tuples
            instances: list of (problem, name) tuples
    """
    instance_order = []
    instance_set = set()
    for r in rows:
        key = (r["problem"], r["name"])
        if key not in instance_set:
            instance_set.add(key)
            instance_order.append(key)

    configs = sorted(set((r["solver"], int(r["cores"])) for r in rows))

    instance_idx = {k: i for i, k in enumerate(instance_order)}
    config_idx = {k: i for i, k in enumerate(configs)}
    n_configs = len(configs)
    n_instances = len(instance_order)

    # Per-(config, instance) data
    status = [[None] * n_instances for _ in range(n_configs)]
    time_ms = [[0.0] * n_instances for _ in range(n_configs)]
    objective = [[None] * n_instances for _ in range(n_configs)]
    wrong = [[False] * n_instances for _ in range(n_configs)]

    # Model per instance (for problem_types lookup)
    instance_model = {}

    for r in rows:
        ci = config_idx[(r["solver"], int(r["cores"]))]
        ii = instance_idx[(r["problem"], r["name"])]
        status[ci][ii] = r["status"]
        time_ms[ci][ii] = float(r["time_ms"])
        objective[ci][ii] = _parse_obj(r.get("objective"))
        wrong[ci][ii] = r.get("wrong", "False") == "True" or r.get("wrong") is True
        instance_model[(r["problem"], r["name"])] = r["model"]

    kind = [
        problem_types.get((p, instance_model.get((p, n), "")), "SAT")
        for p, n in instance_order
    ]

    scores = np.zeros((n_configs, n_instances))

    is_opponent = [opponents is None or c in opponents for c in configs]

    for ii in range(n_instances):
        for ai in range(n_configs):
            for bi in range(ai + 1, n_configs):
                if not is_opponent[ai] and not is_opponent[bi]:
                    continue

                a_broken = wrong[ai][ii] or status[ai][ii] == "Error"
                b_broken = wrong[bi][ii] or status[bi][ii] == "Error"

                if a_broken and b_broken:
                    sa, sb = 0.0, 0.0
                elif a_broken:
                    sa, sb = 0.0, 1.0
                elif b_broken:
                    sa, sb = 1.0, 0.0
                else:
                    sa, sb = _compare(
                        status[ai][ii], time_ms[ai][ii], objective[ai][ii],
                        status[bi][ii], time_ms[bi][ii], objective[bi][ii],
                        kind[ii],
                    )

                if is_opponent[bi]:
                    scores[ai, ii] += sa
                if is_opponent[ai]:
                    scores[bi, ii] += sb

    return scores, configs, instance_order
