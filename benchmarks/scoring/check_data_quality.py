"""
Comprehensive data quality checks on combined.csv and solver .out files.

Checks:
  1. Non-empty .out file but status=Unknown and no objective (parser missed a solution)
  2. Duplicate rows (same solver/cores/year/problem/model/name)
  3. Weird time values (missing, <= 0, or way above 20 min)
  4. Non-numeric objective values
  5. Missing rows (solver has most instances in a year but missing some)
  6. Time < 20 min but no objective (excluding Unsat and SAT problems)
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent / 'open-category-benchmarks'
COMBINED_CSV = ROOT / 'combined.csv'
SOLVERS_DIR = ROOT / 'solvers'
TYPES_CSV = ROOT / 'problem_types.csv'
TIMEOUT_MS = 1_200_000
HIGH_TIME_MS = 1_500_000  # 25 min — suspiciously above timeout


def load_problem_types(path: Path) -> dict[tuple[str, str], str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row['problem'], row['model'])] = row['type']
    return types


def out_filename(row: dict) -> str:
    return (
        f"{row['problem']}-sep-{row['model']}-sep-{row['name']}"
        f"-sep-{row['solver']}-sep-{row['cores']}-sep-0.out"
    )


def fmt(row: dict) -> str:
    parts = []
    if 'year' in row:
        parts.append(row['year'])
    parts.extend([
        f"{row['problem']}/{row['name']}",
        f"model={row['model']}",
        f"solver={row['solver']}",
        f"cores={row['cores']}",
        f"time_ms={row.get('time_ms', '??')}",
        f"status={row['status']}",
        f"obj={row.get('objective', '') or '-'}",
    ])
    return '  '.join(parts)


def main():
    problem_types = load_problem_types(TYPES_CSV)

    with open(COMBINED_CSV) as f:
        rows = list(csv.DictReader(f))

    # ── 1. Non-empty .out file but Unknown/no objective ──────────────────────
    check1 = []
    for results_csv in sorted(SOLVERS_DIR.rglob('results.csv')):
        run_dir = results_csv.parent
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'solver' not in reader.fieldnames:
                continue
            for row in reader:
                if row['status'] != 'Unknown':
                    continue
                obj = row.get('objective', '').strip()
                if obj:
                    continue
                out_path = run_dir / out_filename(row)
                if not out_path.exists():
                    continue
                if out_path.stat().st_size > 0:
                    check1.append((out_path, row))

    print(f'=== 1. Non-empty .out but Unknown with no objective ({len(check1)}) ===')
    for out_path, row in check1[:30]:
        rel = out_path.relative_to(SOLVERS_DIR)
        size = out_path.stat().st_size
        print(f'  {fmt(row)}  file={rel} ({size} bytes)')
    if len(check1) > 30:
        print(f'  ... and {len(check1) - 30} more')
    print()

    # ── 2. Duplicate rows ────────────────────────────────────────────────────
    seen = Counter()
    for row in rows:
        key = (row['solver'], row['cores'], row.get('year', ''),
               row['problem'], row['model'], row['name'])
        seen[key] += 1
    dupes = {k: v for k, v in seen.items() if v > 1}

    print(f'=== 2. Duplicate rows ({len(dupes)}) ===')
    for (solver, cores, year, problem, model, name), count in sorted(dupes.items()):
        print(f'  {year} {problem}/{name} model={model} solver={solver} cores={cores} x{count}')
    print()

    # ── 3. Weird time values ─────────────────────────────────────────────────
    missing_time = []
    zero_or_neg = []
    way_over = []
    for row in rows:
        t = row.get('time_ms', '').strip()
        if not t:
            missing_time.append(row)
            continue
        try:
            t_val = int(t)
        except ValueError:
            missing_time.append(row)
            continue
        if t_val <= 0:
            zero_or_neg.append(row)
        elif t_val > HIGH_TIME_MS:
            way_over.append(row)

    print(f'=== 3a. Missing or non-numeric time_ms ({len(missing_time)}) ===')
    for row in missing_time[:20]:
        print(f'  {fmt(row)}')
    print()

    print(f'=== 3b. time_ms <= 0 ({len(zero_or_neg)}) ===')
    for row in zero_or_neg[:20]:
        print(f'  {fmt(row)}')
    print()

    print(f'=== 3c. time_ms > {HIGH_TIME_MS} ({len(way_over)}) ===')
    for row in sorted(way_over, key=lambda r: -int(r['time_ms']))[:20]:
        print(f'  {fmt(row)}')
    if len(way_over) > 20:
        print(f'  ... and {len(way_over) - 20} more')
    print()

    # ── 4. Non-numeric objective ─────────────────────────────────────────────
    bad_obj = []
    for row in rows:
        obj = row.get('objective', '').strip()
        if not obj:
            continue
        try:
            float(obj)
        except ValueError:
            bad_obj.append(row)

    print(f'=== 4. Non-numeric objective ({len(bad_obj)}) ===')
    for row in bad_obj[:20]:
        print(f'  {fmt(row)}')
    print()

    # ── 5. Missing rows per solver/year ──────────────────────────────────────
    # For each (solver, cores, year), count instances. Flag if significantly below peers.
    instances_per_year: dict[str, set] = defaultdict(set)
    solver_instances: dict[tuple, set] = defaultdict(set)
    for row in rows:
        year = row.get('year', '')
        inst = (row['problem'], row['model'], row['name'])
        instances_per_year[year].add(inst)
        solver_instances[(row['solver'], row['cores'], year)].add(inst)

    print(f'=== 5. Missing rows (solver has < 90% of instances for its year) ===')
    missing_entries = []
    for (solver, cores, year), insts in sorted(solver_instances.items()):
        total = len(instances_per_year[year])
        count = len(insts)
        if total > 0 and count < total * 0.9:
            pct = count / total * 100
            missing = total - count
            missing_entries.append((solver, cores, year, count, total, missing, pct))
    for solver, cores, year, count, total, missing, pct in missing_entries:
        print(f'  {solver:40s} cores={cores}  year={year}  has={count}/{total} ({pct:.0f}%)  missing={missing}')
    print()

    # ── 6. Time < 20 min but no objective (excl. Unsat and SAT) ──────────────
    check6 = []
    for row in rows:
        try:
            time_ms = int(row['time_ms'])
        except (KeyError, ValueError):
            continue
        if time_ms >= TIMEOUT_MS:
            continue
        if row['status'] == 'Unsat':
            continue
        kind = problem_types.get((row['problem'], row['model']))
        if kind == 'SAT':
            continue
        obj = row.get('objective', '').strip()
        if not obj:
            check6.append(row)

    print(f'=== 6. MIN/MAX with time < 20min and no objective (excl. Unsat) ({len(check6)}) ===')
    for row in check6[:30]:
        print(f'  {fmt(row)}')
    if len(check6) > 30:
        print(f'  ... and {len(check6) - 30} more')
    print()


if __name__ == '__main__':
    main()
