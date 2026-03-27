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


def main():
    all_rows = []

    for path in sorted(BASE.glob('dexter8/*.csv')):
        rows = read_dexter(path)
        print(f'dexter8/{path.name}: {len(rows)} rows')
        all_rows.extend(rows)

    for path in sorted(BASE.glob('maybe-better-schedule8/*.csv')):
        rows = read_maybe_better_schedule(path)
        print(f'maybe-better-schedule8/{path.name}: {len(rows)} rows')
        all_rows.extend(rows)

    mzn_csv = BASE / 'allesio' / 'mzn-challenge.csv'
    rows = read_mzn_challenge(mzn_csv)
    print(f'allesio/mzn-challenge.csv: {len(rows)} rows')
    all_rows.extend(rows)

    print(f'\nTotal: {len(all_rows)} rows')

    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'Written to {OUT_PATH}')


if __name__ == '__main__':
    main()
