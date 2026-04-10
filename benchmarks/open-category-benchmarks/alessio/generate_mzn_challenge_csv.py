"""
Reads an extracted mzn_challenge_results folder and produces
a single CSV: mzn-challenge.csv with columns:
  solver, cores, year, name, model, time_ms, objective, status
"""
import json
import re
from pathlib import Path
import csv

RESULTS_PATH = Path('/Users/sofus/speciale/ai/mzn_challenge_results')
DATA_PATH = Path('/Users/sofus/speciale/ai/data')
JSON_2023_PATH = Path(__file__).parent / '2023_results.json'
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
    """Returns {instance_key: (year, model, instance_name, problem)}"""
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
                        key_map[key] = (year, m.stem, i.stem, prob.name)
            else:
                for m in models:
                    key = m.stem + '_'
                    key_map[key] = (year, m.stem, '', prob.name)
    return key_map


def default_objective_regex(problem: str) -> str | None:
    """Returns the regex to extract the objective from default/raw output for a given problem."""
    if 'monitor' in problem:
        return r'monitors\s*=\s*([\-0-9]+)'
    if problem == 'harmony':
        return r'objective:\s*([\-0-9]+)'
    if problem == 'hitori':
        return r'obj\s*=\s*([\-0-9]+)'
    if problem == 'black-hole':
        return None  # SAT problem, no objective
    return r'\bobjective\s*=\s*([\-0-9]+)'


def parse_out_file(path: Path, problem: str = ''):
    """Returns (time_ms, objective, status) from .out file."""
    lines = path.read_text(encoding='utf-8', errors='replace').strip().split('\n')
    time_ms = None
    objective = None
    status = None
    last_solution_objective = None
    warned_multi_source = False

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

            obj_from_json = None
            json_out = out.get('json')
            if json_out:
                try:
                    obj_data = json.loads(json_out) if isinstance(json_out, str) else json_out
                    if '_objective' in obj_data:
                        obj_from_json = obj_data['_objective']
                except (json.JSONDecodeError, TypeError):
                    pass

            obj_from_dzn = None
            dzn_out = out.get('dzn')
            if dzn_out:
                m = re.search(r'_objective\s*=\s*(-?[\d.]+)\s*;', dzn_out)
                if m:
                    obj_from_dzn = float(m.group(1))

            obj_from_default = None
            default_out = out.get('default') or out.get('raw')
            if default_out:
                pattern = default_objective_regex(problem)
                if pattern:
                    m = re.search(pattern, default_out)
                    if m:
                        obj_from_default = float(m.group(1))

            found = {k: v for k, v in [('json', obj_from_json), ('dzn', obj_from_dzn), ('default', obj_from_default)] if v is not None}
            if len(found) > 1:
                vals = set(float(v) for v in found.values())
                if len(vals) > 1 and not warned_multi_source:
                    print(f'WARNING: disagreeing objective sources in {path.name}: {found}')
                    warned_multi_source = True

            if obj_from_json is not None:
                last_solution_objective = obj_from_json
            elif obj_from_dzn is not None:
                last_solution_objective = obj_from_dzn
            elif obj_from_default is not None:
                last_solution_objective = obj_from_default
        elif t == 'status':
            raw_status = d.get('status', '')
            status = STATUS_MAP.get(raw_status, raw_status)
            time_ms = d.get('time')

    objective = last_solution_objective
    if time_ms is None:
        time_ms = 1200000
    if status is None:
        status = 'Unknown'
    return time_ms, objective, status


def main():
    print('Building key map from problem folders...')
    key_map = build_key_map()
    print(f'  {len(key_map)} keys across {len(YEAR_FOLDERS)} year folders')

    print(f'Reading results folder: {RESULTS_PATH}')
    rows = []
    skipped = 0

    out_files = list(RESULTS_PATH.glob('*.out'))
    print(f'  {len(out_files)} .out files found')

    for file_path in out_files:
        parts = file_path.name.split('-sep-')
        if len(parts) < 3:
            skipped += 1
            continue

        instance_key = parts[0]
        solver = parts[1]
        cores = parts[2].replace('.out', '')

        if instance_key not in key_map:
            skipped += 1
            continue

        year, model, instance_name, problem = key_map[instance_key]

        time_ms, objective, status = parse_out_file(file_path, problem)

        rows.append({
            'solver': solver,
            'cores': cores,
            'year': year,
            'problem': problem,
            'model': model,
            'name': instance_name,
            'time_ms': time_ms,
            'objective': objective,
            'status': status,
        })

    print(f'  {len(rows)} rows extracted, {skipped} files skipped')

    print(f'Writing {OUT_PATH}')
    with open(OUT_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['solver', 'cores', 'year', 'problem', 'model', 'name', 'time_ms', 'objective', 'status'])
        writer.writeheader()
        writer.writerows(rows)

    print('Done.')


if __name__ == '__main__':
    main()
