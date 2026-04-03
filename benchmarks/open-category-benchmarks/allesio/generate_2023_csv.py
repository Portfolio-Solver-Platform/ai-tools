"""
Reads 2023_results.json and produces mzn-challenge-2023.csv with columns:
  solver, cores, year, problem, name, model, time_ms, objective, status
"""
import json
import csv
from pathlib import Path

JSON_PATH = Path(__file__).parent / '2023_results.json'
DATA_PATH = Path('/Users/sofus/speciale/ai/data/mznc2023_probs')
OUT_PATH = Path(__file__).parent / 'mzn-challenge-2023.csv'

FIELDNAMES = ['solver', 'cores', 'year', 'problem', 'name', 'model', 'time_ms', 'objective', 'status']


def build_mzn_stem_to_problem() -> dict[str, str]:
    mapping = {}
    for prob in DATA_PATH.iterdir():
        if prob.is_dir():
            for f in prob.iterdir():
                if f.suffix == '.mzn':
                    mapping[f.stem] = prob.name
    return mapping


def main():
    stem_to_problem = build_mzn_stem_to_problem()
    data = json.load(open(JSON_PATH))

    rows = []
    for solver, cores_dict in data.items():
        for cores, entries in cores_dict.items():
            for entry in entries:
                model = entry['model']
                problem = stem_to_problem.get(model, model)

                if entry['optimal'] == 'Optimal':
                    status = 'Optimal'
                elif entry['optimal'] == 'Unsat':
                    status = 'Unsatisfiable'
                elif entry['has_solution']:
                    status = 'Satisfied'
                else:
                    status = 'Unknown'

                # For non-optimal results the JSON 'time' is when the last
                # solution was found, not wall-clock runtime.  The solver
                # would have kept running until the 1200s timeout.
                if status in ('Optimal', 'Unsatisfiable'):
                    time_ms = entry['time']
                else:
                    time_ms = 1200000

                rows.append({
                    'solver': solver,
                    'cores': cores,
                    'year': '2023',
                    'problem': problem,
                    'model': model,
                    'name': entry['name'],
                    'time_ms': time_ms,
                    'objective': entry['objective'],
                    'status': status,
                })

    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f'{len(rows)} rows written to {OUT_PATH}')


if __name__ == '__main__':
    main()
