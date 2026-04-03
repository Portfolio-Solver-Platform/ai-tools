"""
Combines all benchmark CSVs into a single combined.csv with columns:
  solver, cores, year, problem, name, model, time_ms, objective, status
"""
import csv
import re
from pathlib import Path

BASE = Path(__file__).parent
OUT_PATH = BASE / 'combined.csv'
FIELDNAMES = ['solver', 'cores', 'year', 'problem', 'name', 'model', 'time_ms', 'objective', 'status']


def year_from_filename(path: Path) -> str:
    m = re.search(r'(20\d\d)', path.stem)
    return m.group(1) if m else ''


def cores_from_foldername(path: Path) -> str:
    m = re.search(r'(\d+)$', path.parent.name)
    return m.group(1) if m else ''


def read_dexter(path: Path) -> list[dict]:
    year = year_from_filename(path)
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'solver':    row['solver'],
                'cores':     row['cores'],
                'year':      year,
                'problem':   row['problem'],
                'name':      row['name'],
                'model':     row['model'],
                'time_ms':   row['time_ms'],
                'objective': row['objective'],
                'status':    row['status'],
            })
    return rows


def read_maybe_better_schedule(path: Path) -> list[dict]:
    year = year_from_filename(path)
    cores = cores_from_foldername(path)
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'solver':    row['schedule'],
                'cores':     cores,
                'year':      year,
                'problem':   row['problem'],
                'name':      row['name'],
                'model':     row['model'],
                'time_ms':   row['time_ms'],
                'objective': row['objective'],
                'status':    row['optimal'],
            })
    return rows


def read_mzn_challenge(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                'solver':    row['solver'],
                'cores':     row['cores'],
                'year':      row['year'],
                'problem':   row['problem'],
                'name':      row['name'],
                'model':     row['model'],
                'time_ms':   row['time_ms'],
                'objective': row['objective'],
                'status':    row['status'],
            })
    return rows


SOURCES = [
    # (glob_pattern, reader_function_name)
    # 'dexter' reader: has solver, cores, problem, name, model, time_ms, objective, status columns
    # 'parasol' reader: has schedule, problem, name, model, time_ms, objective, optimal columns
    #                    cores inferred from folder name suffix, year from filename
    ('dexter8/*.csv',                'dexter'),
    ('maybe-better-schedule8/*.csv', 'parasol'),
    ('svm-no-static8/*.csv',          'parasol'),
    ('svm8/*.csv',          'parasol'),
]

READERS = {
    'dexter': read_dexter,
    'parasol': read_maybe_better_schedule,
}

EXCLUDED_PROBLEMS = {'travelling-thief'}

STATUS_NORMALIZE = {
    'Unsatisfiable': 'Unsat',
}


def main():
    all_rows = []

    for glob_pattern, reader_name in SOURCES:
        reader = READERS[reader_name]
        for path in sorted(BASE.glob(glob_pattern)):
            rows = reader(path)
            print(f'{path.relative_to(BASE)}: {len(rows)} rows')
            all_rows.extend(rows)

    for name in ('mzn-challenge.csv', 'mzn-challenge-2023.csv'):
        mzn_csv = BASE / 'allesio' / name
        if mzn_csv.exists():
            rows = read_mzn_challenge(mzn_csv)
            print(f'allesio/{name}: {len(rows)} rows')
            all_rows.extend(rows)

    all_rows = [r for r in all_rows if r['problem'] not in EXCLUDED_PROBLEMS]

    for r in all_rows:
        r['status'] = STATUS_NORMALIZE.get(r['status'], r['status'])

    print(f'\nTotal: {len(all_rows)} rows')

    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'Written to {OUT_PATH}')


if __name__ == '__main__':
    main()
