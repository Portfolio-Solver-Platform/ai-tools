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


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row['problem'], row['model'])] = row['type']
    return types


def load_wrong_results(path: Path) -> set[tuple[str, str, str, str, str, str]]:
    """Load the set of (solver, cores, year, problem, model, name) that are wrong."""
    wrong = set()
    if not path.exists():
        return wrong
    with open(path) as f:
        for row in csv.DictReader(f):
            wrong.add((row['solver'], row['cores'], row['year'],
                        row['problem'], row['model'], row['name']))
    return wrong


def is_wrong(row: dict, wrong_results: set) -> bool:
    """Check if a row is marked as wrong or has Error status."""
    if row['status'] == 'Error':
        return True
    key = (row['solver'], row['cores'], row.get('year', ''),
           row['problem'], row['model'], row['name'])
    return key in wrong_results


def is_solved(row: dict, kind: str) -> bool:
    if kind == 'SAT':
        return row['status'] in SOLVED_STATUSES
    # OPT: a feasible solution OR a proof of unsatisfiability both count as solved
    return row['objective'] != '' or row['status'] == 'Unsat'


def is_optimal(row: dict) -> bool:
    # For OPT instances, proving Unsat is a complete proof — treat it like Optimal
    return row['status'] in ('Optimal', 'Unsat')


def get_quality(row: dict, kind: str) -> float | None:
    if row['objective'] == '':
        return None
    val = float(row['objective'])
    return -val if kind == 'MIN' else val


def get_time(row: dict) -> float:
    return float(row['time_ms']) if row['time_ms'] != '' else MAX_TIME_MS


def pairwise_score(s: dict, s2: dict, kind: str,
                    wrong_results: set | None = None) -> float:
    """Return the Borda score of s against s2 on one instance."""
    if wrong_results is not None:
        s_wrong = is_wrong(s, wrong_results)
        s2_wrong = is_wrong(s2, wrong_results)
        if s_wrong and s2_wrong:
            return 0.0
        if s_wrong:
            return 0.0
        if s2_wrong:
            return 1.0

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
                    solver_key: str = 'solver',
                    wrong_results: set | None = None) -> dict[str, float]:
    """
    Score all solver configs across instances.

    instances: {instance_key: [row, ...]} where each row has `solver_key`, `status`,
               `objective`, `time_ms`, and `model` fields.
    problem_types: {(problem, model): kind} mapping.
    solver_key: which field identifies the solver/config in each row.
    wrong_results: set of (solver, cores, year, problem, model, name) tuples
                   that are provably wrong and should receive 0 points.

    Returns {solver_label: total_borda_score}.
    """
    from collections import defaultdict
    scores: dict[str, float] = defaultdict(float)
    unknown_types: set[str] = set()

    for inst_key, group in instances.items():
        problem = group[0]['problem']
        model = group[0]['model']
        kind = problem_types.get((problem, model))
        if kind is None:
            unknown_types.add(f'{problem}/{model}')
            continue

        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                score = pairwise_score(s, s2, kind, wrong_results)
                scores[s[solver_key]] += score

    if unknown_types:
        print(f'WARNING: {len(unknown_types)} models not found in problem_types.csv:')
        for m in sorted(unknown_types):
            print(f'  {m}')
        print()

    return dict(scores)
