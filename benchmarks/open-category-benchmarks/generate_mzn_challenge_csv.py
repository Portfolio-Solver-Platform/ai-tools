"""
Reads mzn_challenge_results.zip directly (no extraction) and produces
a single CSV: mzn-challenge.csv with columns:
  solver, cores, year, name, model, time_ms, objective, status
"""
import json
import os
import zipfile
from pathlib import Path
import csv

ZIP_PATH = Path('/Users/sofus/speciale/ai/mzn_challenge_results.zip')
DATA_PATH = Path('/Users/sofus/speciale/ai/data')
OUT_PATH = Path(__file__).parent / 'mzn-challenge.csv'

YEAR_FOLDERS = ['mznc2022_probs', 'mznc2023_probs', 'mznc2024_probs', 'mznc2025_probs']

STATUS_MAP = {
    'OPTIMAL_SOLUTION': 'Optimal',
    'ALL_SOLUTIONS': 'AllSolutions',
    'UNSATISFIABLE': 'Unsatisfiable',
    'SATISFIED': 'Satisfied',
    'UNKNOWN': 'Unknown',
    'ERROR': 'Error',
    'UNBOUNDED': 'Unbounded',
    'UNSAT_OR_UNBOUNDED': 'UnsatOrUnbounded',
}


def build_key_map():
    """Returns {instance_key: (year, model, instance_name)}"""
    key_map = {}
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
                        key = m.stem + '_' + i.stem
                        key_map[key] = (year, m.stem, i.stem)
            else:
                for m in models:
                    key = m.stem + '_'
                    key_map[key] = (year, m.stem, '')
    return key_map


def parse_out_file(content: bytes):
    """Returns (time_ms, objective, status) from .out file content."""
    lines = content.decode('utf-8', errors='replace').strip().split('\n')
    time_ms = None
    objective = None
    status = None
    last_solution_objective = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue

        t = d.get('type')
        if t == 'solution':
            out = d.get('output', {})
            json_out = out.get('json')
            if json_out:
                try:
                    obj_data = json.loads(json_out) if isinstance(json_out, str) else json_out
                    if '_objective' in obj_data:
                        last_solution_objective = obj_data['_objective']
                except (json.JSONDecodeError, TypeError):
                    pass
        elif t == 'status':
            raw_status = d.get('status', '')
            status = STATUS_MAP.get(raw_status, raw_status)
            time_ms = d.get('time')

    objective = last_solution_objective
    return time_ms, objective, status


def main():
    print('Building key map from problem folders...')
    key_map = build_key_map()
    print(f'  {len(key_map)} keys across {len(YEAR_FOLDERS)} year folders')

    print(f'Reading zip: {ZIP_PATH}')
    rows = []
    skipped = 0

    with zipfile.ZipFile(ZIP_PATH) as z:
        out_files = [n for n in z.namelist() if n.endswith('.out')]
        print(f'  {len(out_files)} .out files found')

        for name in out_files:
            basename = name.split('/')[-1]
            parts = basename.split('-sep-')
            if len(parts) < 3:
                skipped += 1
                continue

            instance_key = parts[0]
            solver = parts[1]
            cores = parts[2].replace('.out', '')

            if instance_key not in key_map:
                skipped += 1
                continue

            year, model, instance_name = key_map[instance_key]

            content = z.read(name)
            time_ms, objective, status = parse_out_file(content)

            rows.append({
                'solver': solver,
                'cores': cores,
                'year': year,
                'model': model,
                'name': instance_name,
                'time_ms': time_ms,
                'objective': objective,
                'status': status,
            })

    print(f'  {len(rows)} rows extracted, {skipped} files skipped')

    print(f'Writing {OUT_PATH}')
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['solver', 'cores', 'year', 'model', 'name', 'time_ms', 'objective', 'status'])
        writer.writeheader()
        writer.writerows(rows)

    print('Done.')


if __name__ == '__main__':
    main()
