"""
Computes Borda count scores from a combined benchmark CSV.

Scores are reported per (solver, cores, year).

Rules (complete scoring, MiniZinc challenge style):
  - Each instance is a voter ranking all (solver, cores) combos.
  - For each pair (s, s') on instance i:
      * If not solved(s): s scores 0 (always, even if s' also not solved)
      * If s better than s': s scores 1
      * If s worse than s': s scores 0
      * If indistinguishable (both solved, same quality): s scores
          time(s') / (time(s') + time(s)), or 0.5 if both times are 0
  - solved(s):
      SAT  -> status in {Satisfied, AllSolutions, Unsatisfiable, Optimal}
      OPT  -> objective is not empty (found at least one solution)
  - optimal(s): status == 'Optimal'
  - quality comparison (OPT only):
      MIN  -> lower objective is better
      MAX  -> higher objective is better
  - null time_ms is treated as MAX_TIME_MS (solver was killed mid-run)
"""
import csv
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH       = Path(__file__).parent.parent / 'open-category-benchmarks' / 'combined.csv'
TYPES_CSV_PATH = Path(__file__).parent / 'problem_types.csv'
YEARS          = ['2023', '2024', '2025']   # which years to include
MAX_TIME_MS    = 1_200_000          # fallback when time_ms is missing
# ───────────────────────────────────────────────────────────────────────────────

SOLVED_STATUSES = {'Satisfied', 'AllSolutions', 'Unsatisfiable', 'Optimal'}


def load_problem_types(path: Path) -> dict[str, str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[row['model']] = row['type']
    return types


def is_solved(row: dict, kind: str) -> bool:
    if kind == 'SAT':
        return row['status'] in SOLVED_STATUSES
    else:  # MIN or MAX
        return row['objective'] != ''


def is_optimal(row: dict) -> bool:
    return row['status'] == 'Optimal'


def get_quality(row: dict, kind: str) -> float | None:
    if row['objective'] == '':
        return None
    val = float(row['objective'])
    return -val if kind == 'MIN' else val  # higher is always better


def get_time(row: dict) -> float:
    return float(row['time_ms']) if row['time_ms'] != '' else MAX_TIME_MS


def pairwise_score(s: dict, s2: dict, kind: str) -> float:
    """Return the Borda score of s against s2 on one instance."""
    if not is_solved(s, kind):
        return 0.0

    if kind == 'SAT':
        if not is_solved(s2, kind):
            return 1.0
        # Both solved — indistinguishable, use time
        t_s  = get_time(s)
        t_s2 = get_time(s2)
        if t_s + t_s2 == 0:
            return 0.5
        return t_s2 / (t_s + t_s2)

    # Optimization (MIN or MAX)
    solved_s2 = is_solved(s2, kind)
    if not solved_s2:
        return 1.0

    opt_s  = is_optimal(s)
    opt_s2 = is_optimal(s2)
    if opt_s and not opt_s2:
        return 1.0
    if not opt_s and opt_s2:
        return 0.0

    q_s  = get_quality(s,  kind)
    q_s2 = get_quality(s2, kind)

    if q_s is not None and q_s2 is not None:
        if q_s > q_s2:
            return 1.0
        if q_s < q_s2:
            return 0.0

    # Indistinguishable — use time
    t_s  = get_time(s)
    t_s2 = get_time(s2)
    if t_s + t_s2 == 0:
        return 0.5
    return t_s2 / (t_s + t_s2)


def main():
    problem_types = load_problem_types(TYPES_CSV_PATH)

    rows = list(csv.DictReader(open(CSV_PATH)))
    rows = [r for r in rows if r['year'] in YEARS]

    # Group by instance
    instances: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r['year'], r['problem'], r['model'], r['name'])
        instances[key].append(r)

    # scores[(solver, cores, year)] = total Borda points
    scores: dict[tuple, float] = defaultdict(float)

    unknown_types = set()
    for (year, problem, model, name), group in instances.items():
        kind = problem_types.get(model)
        if kind is None:
            unknown_types.add(model)
            continue

        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                score = pairwise_score(s, s2, kind)
                key = (s['solver'], s['cores'], year)
                scores[key] += score

    if unknown_types:
        print(f'WARNING: {len(unknown_types)} models not found in problem_types.csv:')
        for m in sorted(unknown_types):
            print(f'  {m}')
        print()

    # Print results grouped by year
    for year in YEARS:
        year_scores = {(solver, cores): v
                       for (solver, cores, y), v in scores.items() if y == year}
        print(f'Year: {year}')
        print(f'  {"solver":<30} {"cores":>5}  {"score":>10}')
        print(f'  {"-"*30}  {"-"*5}  {"-"*10}')
        for (solver, cores), score in sorted(year_scores.items(), key=lambda x: -x[1]):
            print(f'  {solver:<30} {cores:>5}  {score:>10.2f}')
        print()


if __name__ == '__main__':
    main()
