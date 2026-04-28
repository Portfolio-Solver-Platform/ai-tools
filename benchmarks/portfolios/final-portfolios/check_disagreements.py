"""
Find instances where the 3 portfolios disagree on results that should be
consistent across runs of the same instance:
  - One portfolio says Unsat, another finds a solution
  - Two portfolios both claim Optimal but with different objectives

Either disagreement points at a wrong-result bug somewhere — almost certainly
in one of the underlying solvers in the disagreeing portfolio's schedule.
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'portfolios-final'
TYPES_CSV = Path(__file__).resolve().parent.parent.parent / 'open-category-benchmarks' / 'problem_types.csv'
PORTFOLIOS = ['cpsat8', 'k1-8c-8s-v1', 'ek1-8c-8s-v2']

SOLVED_STATUSES = {'Satisfied', 'Optimal'}  # exclude 'Unsat' here — opposite case


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row['problem'], row['model'])] = row['type']
    return types


def normalize_status(row: dict) -> str:
    raw = row.get('optimal', '')
    if raw == 'Unknown' and row.get('objective', '').strip():
        return 'Satisfied'
    return raw


def main():
    problem_types = load_problem_types(TYPES_CSV)
    instances: dict[tuple, list[dict]] = defaultdict(list)

    for portfolio in PORTFOLIOS:
        for cfg in sorted((ROOT / portfolio).iterdir()):
            if not cfg.is_dir(): continue
            year = cfg.name.split('-')[-1]
            with open(cfg / 'results.csv') as f:
                for r in csv.DictReader(f):
                    r['__portfolio'] = portfolio
                    r['__year'] = year
                    r['__status_norm'] = normalize_status(r)
                    instances[(year, r['problem'], r['model'], r['name'])].append(r)

    unsat_vs_solved = []
    optimal_disagree = []

    for key, group in instances.items():
        year, problem, model, name = key
        kind = problem_types.get((problem, model))

        unsats = [r for r in group if r['__status_norm'] == 'Unsat']
        solveds = [r for r in group if r['__status_norm'] in SOLVED_STATUSES]
        if unsats and solveds:
            unsat_vs_solved.append((key, unsats, solveds))

        if kind in ('MIN', 'MAX'):
            opts = [r for r in group
                    if r['__status_norm'] == 'Optimal' and r.get('objective', '').strip()]
            if len(opts) >= 2:
                if len({float(r['objective']) for r in opts}) > 1:
                    optimal_disagree.append((key, opts))

    print(f'=== UNSAT vs SOLVED disagreements ({len(unsat_vs_solved)}) ===\n')
    for (year, problem, name, model), unsats, solveds in sorted(unsat_vs_solved):
        print(f'{year} {problem}/{name} (model={model})')
        for r in unsats:
            print(f'  UNSAT:  portfolio={r["__portfolio"]:<14} time={r["time_ms"]:>9}  '
                  f'last_result_from={r.get("last_result_from","")}')
        for r in solveds:
            obj = r.get('objective', '') or '-'
            print(f'  SOLVED: portfolio={r["__portfolio"]:<14} time={r["time_ms"]:>9}  '
                  f'obj={obj:<12} status={r["__status_norm"]:<10} '
                  f'last_result_from={r.get("last_result_from","")}')
        print()

    print(f'\n=== OPTIMAL objective disagreements ({len(optimal_disagree)}) ===\n')
    for (year, problem, name, model), opts in sorted(optimal_disagree):
        kind = problem_types.get((problem, model), '?')
        print(f'{year} {problem}/{name} (model={model}, type={kind})')
        for r in sorted(opts, key=lambda r: float(r['objective'])):
            print(f'  portfolio={r["__portfolio"]:<14} obj={r["objective"]:<12} '
                  f'time={r["time_ms"]:>9}  last_result_from={r.get("last_result_from","")}')
        print()


if __name__ == '__main__':
    main()
