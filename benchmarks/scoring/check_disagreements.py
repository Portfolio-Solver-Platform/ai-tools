"""
Finds instances where solvers disagree on results that should be consistent:
  - One solver says Unsatisfiable, another finds a solution
  - Two solvers both claim Optimal but have different objectives
"""
import csv
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / 'open-category-benchmarks' / 'combined.csv'
TYPES_CSV_PATH = Path(__file__).parent / 'problem_types.csv'

SOLVED_STATUSES = {'Satisfied', 'Optimal', 'AllSolutions'}


def load_problem_types(path: Path) -> dict[str, str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[row['model']] = row['type']
    return types


def main():
    problem_types = load_problem_types(TYPES_CSV_PATH)

    instances: dict[tuple, list[dict]] = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            key = (row['year'], row['problem'], row['model'], row['name'])
            instances[key].append(row)

    unsat_vs_solved = []
    optimal_disagree = []

    for (year, problem, model, name), group in instances.items():
        kind = problem_types.get(model)

        has_unsat = [r for r in group if r['status'] in ('Unsatisfiable', 'Unsat')]
        has_solved = [r for r in group if r['status'] in SOLVED_STATUSES]

        if has_unsat and has_solved:
            unsat_vs_solved.append((year, problem, name, model, has_unsat, has_solved))

        if kind in ('MIN', 'MAX'):
            optimals = [r for r in group if r['status'] == 'Optimal' and r['objective']]
            if len(optimals) >= 2:
                objs = set(float(r['objective']) for r in optimals)
                if len(objs) > 1:
                    optimal_disagree.append((year, problem, name, model, optimals))

    print(f'=== UNSAT vs SOLVED disagreements ({len(unsat_vs_solved)}) ===\n')
    for year, problem, name, model, unsats, solveds in sorted(unsat_vs_solved):
        print(f'{year} {problem}/{name} (model={model})')
        for r in unsats:
            print(f'  UNSAT:   {r["solver"]:>30} cores={r["cores"]}  time={r["time_ms"]}')
        for r in solveds:
            obj = r['objective'] or '-'
            print(f'  SOLVED:  {r["solver"]:>30} cores={r["cores"]}  time={r["time_ms"]}  obj={obj}  status={r["status"]}')
        print()

    print(f'=== OPTIMAL objective disagreements ({len(optimal_disagree)}) ===\n')
    for year, problem, name, model, optimals in sorted(optimal_disagree):
        print(f'{year} {problem}/{name} (model={model}, type={problem_types.get(model)})')
        for r in sorted(optimals, key=lambda r: float(r['objective'])):
            print(f'  {r["solver"]:>30} cores={r["cores"]}  obj={r["objective"]}  time={r["time_ms"]}')
        print()


if __name__ == '__main__':
    main()
