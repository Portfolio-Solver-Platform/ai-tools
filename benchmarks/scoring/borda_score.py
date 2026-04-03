"""
Computes Borda count scores from a combined benchmark CSV.

Scores are reported per (solver, cores, year).
Uses shared scoring logic from borda.py.
"""
import csv
from collections import defaultdict
from pathlib import Path

from borda import load_problem_types, pairwise_score

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH       = Path(__file__).parent.parent / 'open-category-benchmarks' / 'combined.csv'
TYPES_CSV_PATH = Path(__file__).parent / 'problem_types.csv'
YEARS          = ['2023', '2024', '2025']
MAX_TIME_MS    = 1_200_000          # fallback when time_ms is missing
# ───────────────────────────────────────────────────────────────────────────────


def main():
    import borda
    borda.MAX_TIME_MS = MAX_TIME_MS
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
