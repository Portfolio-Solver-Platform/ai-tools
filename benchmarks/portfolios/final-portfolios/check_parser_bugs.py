"""
Mirror of benchmarks/scoring/check_parser_bugs.py adapted to final-portfolios/.
Looks for rows whose (status, objective, problem-type) combinations can only
come from a parser/harness bug — not from a solver actually behaving that way.

CSV column quirk vs. open-cat: our results.csv uses `optimal` for what the
open-cat checks call `status`, and `Satisfied` is encoded by `optimal=Unknown`
combined with a non-empty `objective`. We normalize on read so the rest of
this script is identical to the open-cat one.

Checks (semantic, on each results.csv):
  A. status == Unsat but objective is non-empty
  B. status == Error but objective is non-empty
  C. (MIN/MAX) status == Satisfied but objective is empty
  D. (MIN/MAX) status == Optimal  but objective is empty
  E. (SAT)     objective is non-empty
  F. (SAT)     status == Optimal

Checks (.out files alongside, requires harness convention):
  G. .out file is empty but objective is non-empty
  H. .out file is empty but status is not Unknown / Satisfied
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'portfolios-final'
TYPES_CSV = Path(__file__).resolve().parent.parent.parent / 'open-category-benchmarks' / 'problem_types.csv'


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row['problem'], row['model'])] = row['type']
    return types


def has_obj(row: dict) -> bool:
    return bool(row.get('objective', '').strip())


def normalize(row: dict) -> dict:
    """Map our CSV's `optimal` field onto the `status` semantics the checks
    were written against (i.e. include the `Unknown + obj => Satisfied` recovery
    that combine_benchmarks.py also applies)."""
    out = dict(row)
    raw = row.get('optimal', '')
    if raw == 'Unknown' and has_obj(row):
        out['status'] = 'Satisfied'
    else:
        out['status'] = raw
    return out


def out_filename(row: dict) -> str:
    schedule = row.get('schedule', '')
    return (
        f"{row['problem']}-sep-{row['model']}-sep-{row['name']}"
        f"-sep-{schedule}-sep-8-sep-0.out"
    )


def fmt(row: dict) -> str:
    return '  '.join([
        f"{row.get('schedule', '?')}",
        f"{row['problem']}/{row['name']}",
        f"model={row['model']}",
        f"time_ms={row['time_ms']}",
        f"status={row['status']}",
        f"obj={row.get('objective', '') or '-'}",
    ])


def main():
    problem_types = load_problem_types(TYPES_CSV)
    buckets: dict[str, list[dict]] = {k: [] for k in 'ABCDEFGH'}

    for results_csv in sorted(ROOT.rglob('results.csv')):
        run_dir = results_csv.parent
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'optimal' not in reader.fieldnames:
                continue
            for raw_row in reader:
                row = normalize(raw_row)
                row['__cfg'] = run_dir.name
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

                # .out file checks
                out_path = run_dir / out_filename(raw_row)
                if out_path.exists() and out_path.stat().st_size == 0:
                    enriched = dict(row)
                    enriched['__out'] = str(out_path.relative_to(ROOT))
                    if has_obj(row):
                        buckets['G'].append(enriched)
                    if row['status'] not in ('Unknown',):
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
        for row in rows[:40]:
            line = fmt(row) + f"  cfg={row['__cfg']}"
            if '__out' in row:
                line += f"  file={row['__out']}"
            print('  ' + line)
        if len(rows) > 40:
            print(f'  ... and {len(rows) - 40} more')
        print()


if __name__ == '__main__':
    main()
