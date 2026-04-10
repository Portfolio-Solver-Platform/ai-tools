"""
Computes Borda count scores from a combined benchmark CSV.

Scores are reported per (solver, cores, year).
Uses shared scoring logic from borda.py.
"""
import csv
from collections import defaultdict
from pathlib import Path

from borda import load_problem_types, load_wrong_results, pairwise_score

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH       = Path(__file__).parent.parent / 'open-category-benchmarks' / 'combined.csv'
TYPES_CSV_PATH = Path(__file__).parent.parent / 'open-category-benchmarks' / 'problem_types.csv'
WRONG_CSV_PATH = Path(__file__).parent.parent / 'open-category-benchmarks' / 'wrong_results.csv'
YEARS          = ['2023', '2024', '2025']
MAX_TIME_MS    = 1_200_000          # fallback when time_ms is missing
# ───────────────────────────────────────────────────────────────────────────────


def compute_scores(rows: list[dict], problem_types: dict[str, str],
                   wrong_results: set) -> tuple[dict, set]:
    instances: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r['year'], r['problem'], r['model'], r['name'])
        instances[key].append(r)

    scores: dict[tuple, float] = defaultdict(float)
    unknown_types: set[str] = set()
    for (year, problem, model, name), group in instances.items():
        kind = problem_types.get((problem, model))
        if kind is None:
            unknown_types.add(f'{problem}/{model}')
            continue
        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                scores[(s['solver'], s['cores'], year)] += pairwise_score(s, s2, kind, wrong_results)
    return scores, unknown_types


def print_scores(label: str, scores: dict) -> None:
    print(f'═══ {label} ═══')
    for year in YEARS:
        year_scores = {(solver, cores): v
                       for (solver, cores, y), v in scores.items() if y == year}
        if not year_scores:
            continue
        print(f'Year: {year}')
        print(f'  {"solver":<30} {"cores":>5}  {"score":>10}')
        print(f'  {"-"*30}  {"-"*5}  {"-"*10}')
        for (solver, cores), score in sorted(year_scores.items(), key=lambda x: -x[1]):
            print(f'  {solver:<30} {cores:>5}  {score:>10.2f}')
        print()


def main():
    import borda
    borda.MAX_TIME_MS = MAX_TIME_MS
    problem_types = load_problem_types(TYPES_CSV_PATH)
    wrong_results = load_wrong_results(WRONG_CSV_PATH)

    rows = list(csv.DictReader(open(CSV_PATH)))
    rows = [r for r in rows if r['year'] in YEARS]

    # Full score: every solver/config competes against every other on each instance.
    all_scores, unknown_all = compute_scores(rows, problem_types, wrong_results)

    # Open-category score: only each solver's best variant (highest cores per year)
    # competes, matching MiniZinc's "select all open solvers" behavior.
    open_rows = [r for r in rows if r.get('open_category') == 'True']
    open_scores, _ = compute_scores(open_rows, problem_types, wrong_results)

    if unknown_all:
        print(f'WARNING: {len(unknown_all)} models not found in problem_types.csv:')
        for m in sorted(unknown_all):
            print(f'  {m}')
        print()

    print_scores('All configurations', all_scores)
    print_scores('Open category only', open_scores)


if __name__ == '__main__':
    main()
