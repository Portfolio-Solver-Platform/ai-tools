"""
Adds a 'problem' column to mzn-challenge.csv by looking up which subfolder
in the data directories contains the model file for each row.
"""
import csv
from pathlib import Path

DATA_PATH = Path('/Users/sofus/speciale/ai/data')
CSV_PATH = Path(__file__).parent / 'mzn-challenge.csv'
YEAR_FOLDERS = ['mznc2022_probs', 'mznc2023_probs', 'mznc2024_probs', 'mznc2025_probs']


def build_problem_map():
    """Returns {(year, model, instance_name): problem_folder_name}"""
    problem_map = {}
    for year_folder in YEAR_FOLDERS:
        path = DATA_PATH / year_folder
        if not path.is_dir():
            continue
        year = year_folder.replace('mznc', '').replace('_probs', '')
        for prob in sorted(path.iterdir()):
            if not prob.is_dir():
                continue
            files = list(prob.iterdir())
            models = [f for f in files if f.suffix == '.mzn']
            instances = [f for f in files if f.suffix in ('.dzn', '.json')]
            if instances:
                for m in models:
                    for i in instances:
                        problem_map[(year, m.stem, i.stem)] = prob.name
            else:
                for m in models:
                    problem_map[(year, m.stem, '')] = prob.name
    return problem_map


def main():
    problem_map = build_problem_map()
    print(f'Built problem map with {len(problem_map)} entries')

    with open(CSV_PATH, 'r') as f:
        rows = list(csv.DictReader(f))

    missing = 0
    for row in rows:
        key = (row['year'], row['model'], row['name'])
        problem = problem_map.get(key)
        if problem is None:
            missing += 1
        row['problem'] = problem or ''

    print(f'Rows missing problem lookup: {missing}')

    fieldnames = ['solver', 'cores', 'year', 'problem', 'model', 'name', 'time_ms', 'objective', 'status']
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written {len(rows)} rows with problem column to {CSV_PATH}')


if __name__ == '__main__':
    main()
