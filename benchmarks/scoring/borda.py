from __future__ import annotations
"""
Shared Borda count scoring logic (MiniZinc challenge style).

Each row dict is expected to have keys: status, objective, time_ms.
The `kind` for each instance is one of: SAT, MIN, MAX.
"""

import csv
from pathlib import Path

MAX_TIME_MS = 1_200_000
SOLVED_STATUSES = {'Satisfied', 'AllSolutions', 'Unsat', 'Optimal'}


def load_problem_types(path: Path) -> dict[str, str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[row['model']] = row['type']
    return types


def is_solved(row: dict, kind: str) -> bool:
    if kind == 'SAT':
        return row['status'] in SOLVED_STATUSES
    else:
        return row['objective'] != ''


def is_optimal(row: dict) -> bool:
    return row['status'] == 'Optimal'


def get_quality(row: dict, kind: str) -> float | None:
    if row['objective'] == '':
        return None
    val = float(row['objective'])
    return -val if kind == 'MIN' else val


def get_time(row: dict) -> float:
    return float(row['time_ms']) if row['time_ms'] != '' else MAX_TIME_MS


def pairwise_score(s: dict, s2: dict, kind: str) -> float:
    """Return the Borda score of s against s2 on one instance."""
    if not is_solved(s, kind):
        return 0.0

    if kind == 'SAT':
        if not is_solved(s2, kind):
            return 1.0
        t_s = get_time(s)
        t_s2 = get_time(s2)
        if t_s + t_s2 == 0:
            return 0.5
        return t_s2 / (t_s + t_s2)

    # Optimization (MIN or MAX)
    if not is_solved(s2, kind):
        return 1.0

    opt_s = is_optimal(s)
    opt_s2 = is_optimal(s2)
    if opt_s and not opt_s2:
        return 1.0
    if not opt_s and opt_s2:
        return 0.0

    q_s = get_quality(s, kind)
    q_s2 = get_quality(s2, kind)
    if q_s is not None and q_s2 is not None:
        if q_s > q_s2:
            return 1.0
        if q_s < q_s2:
            return 0.0

    # Indistinguishable — use time
    t_s = get_time(s)
    t_s2 = get_time(s2)
    if t_s + t_s2 == 0:
        return 0.5
    return t_s2 / (t_s + t_s2)


def score_instances(instances: dict[tuple, list[dict]], problem_types: dict[str, str],
                    solver_key: str = 'solver') -> dict[str, float]:
    """
    Score all solver configs across instances.

    instances: {instance_key: [row, ...]} where each row has `solver_key`, `status`,
               `objective`, `time_ms`, and `model` fields.
    problem_types: {model: kind} mapping.
    solver_key: which field identifies the solver/config in each row.

    Returns {solver_label: total_borda_score}.
    """
    from collections import defaultdict
    scores: dict[str, float] = defaultdict(float)
    unknown_types: set[str] = set()

    for inst_key, group in instances.items():
        model = group[0]['model']
        kind = problem_types.get(model)
        if kind is None:
            unknown_types.add(model)
            continue

        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                score = pairwise_score(s, s2, kind)
                scores[s[solver_key]] += score

    if unknown_types:
        print(f'WARNING: {len(unknown_types)} models not found in problem_types.csv:')
        for m in sorted(unknown_types):
            print(f'  {m}')
        print()

    return dict(scores)
