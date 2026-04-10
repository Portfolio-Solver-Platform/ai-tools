"""Analyse which solver finds the best solution fastest, and speed ratios across cores.

Input: benchmarks/open-category-benchmarks/combined.csv (path resolved relative
to this script, with an env-var override for ad-hoc runs).

Outputs three sections:
  Q1 - Winner counts per (solver, cores) and per solver family.
  Q2 - Pairwise speed ratios across cores of the same solver.
  Q3 - Each core config vs the average of its sibling configs.
"""
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

AI_TOOLS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = AI_TOOLS_ROOT / 'benchmarks' / 'open-category-benchmarks' / 'combined.csv'
CSV_PATH = Path(os.environ.get('COMBINED_CSV', DEFAULT_CSV))

df = pd.read_csv(CSV_PATH)

# A row has a "solution" if status is Optimal or Satisfied.
df['has_solution'] = df['status'].isin(['Optimal', 'Satisfied'])
df['config'] = df['solver'] + '-' + df['cores'].astype(str) + 'c'

INSTANCE_KEYS = ['year', 'problem', 'name']


def infer_direction(group: pd.DataFrame) -> str:
    """Return 'min' or 'max' for this instance.

    If any row is Optimal, that objective is the true optimum. Determine
    direction by comparing Satisfied objectives against it:
      Satisfied > Optimal  => minimization
      Satisfied < Optimal  => maximization
    Fall back to minimization when ambiguous (most CP problems).
    """
    opt_rows = group[group['status'] == 'Optimal']
    if len(opt_rows) == 0:
        return 'min'
    opt_vals = opt_rows['objective'].dropna()
    if len(opt_vals) == 0:
        return 'min'
    opt_val = opt_vals.iloc[0]
    sat_rows = group[(group['status'] == 'Satisfied') & group['objective'].notna()]
    if len(sat_rows) == 0:
        return 'min'
    if (sat_rows['objective'] > opt_val).any():
        return 'min'
    if (sat_rows['objective'] < opt_val).any():
        return 'max'
    return 'min'


def best_obj(group: pd.DataFrame, direction: str):
    vals = group[group['has_solution']]['objective'].dropna()
    if len(vals) == 0:
        return None
    return vals.min() if direction == 'min' else vals.max()


def gmean(arr) -> float:
    arr = np.asarray([x for x in arr if x > 0], dtype=float)
    if len(arr) == 0:
        return float('nan')
    return float(np.exp(np.log(arr).mean()))


direction_map = {}
best_map = {}
for key, grp in df.groupby(INSTANCE_KEYS):
    direction = infer_direction(grp)
    direction_map[key] = direction
    best_map[key] = best_obj(grp, direction)

# ---- Q1: winner counts ------------------------------------------------------
# For each instance, find solvers that reached the best objective and pick the
# one with the smallest time_ms. Ties on time_ms credit every tied solver.
wins_config: dict[str, int] = {}
wins_solver: dict[str, int] = {}
instances_with_winner = 0
n_instances = 0
for key, grp in df.groupby(INSTANCE_KEYS):
    n_instances += 1
    best = best_map[key]
    if best is None:
        continue
    ok = grp[grp['has_solution'] & (grp['objective'] == best)]
    if len(ok) == 0:
        continue
    fastest = ok['time_ms'].min()
    winners = ok[ok['time_ms'] == fastest]
    instances_with_winner += 1
    for _, row in winners.iterrows():
        wins_config[row['config']] = wins_config.get(row['config'], 0) + 1
        wins_solver[row['solver']] = wins_solver.get(row['solver'], 0) + 1

print(f'{n_instances} instances, {instances_with_winner} with a winner  ({CSV_PATH.name})')
print()
print('Wins per (solver, cores):')
for cfg, cnt in sorted(wins_config.items(), key=lambda x: -x[1]):
    print(f'  {cnt:4d}  {cfg}')
print()
print('Wins per solver family:')
for s, cnt in sorted(wins_solver.items(), key=lambda x: -x[1]):
    print(f'  {cnt:4d}  {s}')
print()

# ---- Q2: speed ratios across cores of the same solver ----------------------
# Pairwise ratios are restricted to instances where BOTH configs reached the
# SAME objective value (apples-to-apples comparison). Solutions-found counts
# are reported separately so uneven coverage is visible.

solver_cores = df.groupby('solver')['cores'].unique().to_dict()
multi_core_solvers = {s: sorted(c) for s, c in solver_cores.items() if len(c) > 1}

print('Speed across core configs (pairwise, on instances where both reached the same objective):')
for solver, cores_list in multi_core_solvers.items():
    sub = df[df['solver'] == solver]
    pivot_time = sub.pivot_table(index=INSTANCE_KEYS, columns='cores',
                                 values='time_ms', aggfunc='first')
    pivot_sol = sub.pivot_table(index=INSTANCE_KEYS, columns='cores',
                                values='has_solution', aggfunc='first')
    pivot_obj = sub.pivot_table(index=INSTANCE_KEYS, columns='cores',
                                values='objective', aggfunc='first')

    solved_counts = ', '.join(
        f'{c}c={int((pivot_sol[c] == True).sum())}'  # noqa: E712
        for c in cores_list if c in pivot_sol.columns
    )
    print(f'  {solver}  (solutions found: {solved_counts})')
    for a, b in combinations(cores_list, 2):
        if a not in pivot_time.columns or b not in pivot_time.columns:
            continue
        same = (
            (pivot_sol[a] == True) & (pivot_sol[b] == True)  # noqa: E712
            & (pivot_obj[a] == pivot_obj[b])
        )
        n = int(same.sum())
        if n == 0:
            print(f'    {a}c vs {b}c: no overlap')
            continue
        g = gmean(pivot_time.loc[same, a] / pivot_time.loc[same, b])
        pct = (g - 1) * 100
        word = 'slower' if pct > 0 else 'faster'
        print(f'    {a}c vs {b}c (n={n:3d}): {a}c is {abs(pct):5.1f}% {word}')
