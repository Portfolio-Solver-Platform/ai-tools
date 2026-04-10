"""
Looks for rows whose (status, objective, problem-type) combinations can only
come from a parser/harness bug — not from a solver actually behaving that way.

Checks on combined.csv (semantic):
  A. status == Unsat but objective is non-empty
  B. status == Error but objective is non-empty
  C. (MIN/MAX) status == Satisfied but objective is empty
  D. (MIN/MAX) status == Optimal  but objective is empty
  E. (SAT)     objective is non-empty
  F. (SAT)     status == Optimal

Checks on solvers/**/results.csv (need the .out file alongside):
  G. .out file is empty but objective is non-empty
  H. .out file is empty but status is not Unknown
"""
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent / 'open-category-benchmarks'
COMBINED_CSV = ROOT / 'combined.csv'
SOLVERS_DIR = ROOT / 'solvers'
TYPES_CSV = Path(__file__).parent.parent / 'open-category-benchmarks' / 'problem_types.csv'


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row['problem'], row['model'])] = row['type']
    return types


def has_obj(row: dict) -> bool:
    return bool(row.get('objective', '').strip())


def out_filename(row: dict) -> str:
    return (
        f"{row['problem']}-sep-{row['model']}-sep-{row['name']}"
        f"-sep-{row['solver']}-sep-{row['cores']}-sep-0.out"
    )


def fmt(row: dict) -> str:
    extras = []
    if 'year' in row:
        extras.append(row['year'])
    extras.extend([
        f"{row['problem']}/{row['name']}",
        f"model={row['model']}",
        f"solver={row['solver']}",
        f"cores={row['cores']}",
        f"time_ms={row['time_ms']}",
        f"status={row['status']}",
        f"obj={row.get('objective', '') or '-'}",
    ])
    return '  '.join(extras)


def main():
    problem_types = load_problem_types(TYPES_CSV)

    buckets: dict[str, list[dict]] = {k: [] for k in 'ABCDEFGH'}

    # ---- semantic checks on combined.csv ----
    with open(COMBINED_CSV) as f:
        for row in csv.DictReader(f):
            status = row['status']
            kind = problem_types.get((row['problem'], row['model']))
            obj = has_obj(row)

            if status == 'Unsat' and obj:
                buckets['A'].append(row)
            if status == 'Error' and obj:
                buckets['B'].append(row)
            if kind in ('MIN', 'MAX'):
                if status == 'Satisfied' and not obj:
                    buckets['C'].append(row)
                if status == 'Optimal' and not obj:
                    buckets['D'].append(row)
            if kind == 'SAT':
                if obj:
                    buckets['E'].append(row)
                if status == 'Optimal':
                    buckets['F'].append(row)

    # ---- .out file checks ----
    for results_csv in sorted(SOLVERS_DIR.rglob('results.csv')):
        run_dir = results_csv.parent
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'solver' not in reader.fieldnames:
                continue
            for row in reader:
                out_path = run_dir / out_filename(row)
                if not out_path.exists() or out_path.stat().st_size != 0:
                    continue
                # annotate so fmt() can show the path
                enriched = dict(row)
                enriched['__out'] = str(out_path.relative_to(SOLVERS_DIR))
                if has_obj(row):
                    buckets['G'].append(enriched)
                if row['status'] != 'Unknown':
                    buckets['H'].append(enriched)

    headings = {
        'A': 'Unsat with non-empty objective',
        'B': 'Error with non-empty objective',
        'C': 'MIN/MAX Satisfied with empty objective',
        'D': 'MIN/MAX Optimal with empty objective',
        'E': 'SAT row has a non-empty objective',
        'F': 'SAT row has status=Optimal',
        'G': 'Empty .out file with non-empty objective',
        'H': 'Empty .out file with status != Unknown',
    }

    for key in 'ABCDEFGH':
        rows = buckets[key]
        print(f'=== {key}. {headings[key]} ({len(rows)}) ===')
        for row in rows[:50]:
            line = fmt(row)
            if '__out' in row:
                line += f"  file={row['__out']}"
            print('  ' + line)
        if len(rows) > 50:
            print(f'  ... and {len(rows) - 50} more')
        print()


if __name__ == '__main__':
    main()
