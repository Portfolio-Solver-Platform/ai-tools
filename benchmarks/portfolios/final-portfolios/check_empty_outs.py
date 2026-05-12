"""
Mirror of benchmarks/scoring/check_empty_outs.py for final-portfolios/.

Finds entries where the .out file is empty but the recorded time is short,
which usually means a crashed/aborted run masquerading as a real result.

Also flags non-existent .out files (the harness should have written one for
every row it logged).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'portfolios-final'
TIMEOUT_MS = 1_200_000


def out_filename(row: dict) -> str:
    schedule = row['schedule']
    return (
        f"{row['problem']}-sep-{row['model']}-sep-{row['name']}"
        f"-sep-{schedule}-sep-8-sep-0.out"
    )


def main():
    suspicious_empty: list[tuple[Path, dict]] = []
    suspicious_missing: list[tuple[Path, dict]] = []

    for results_csv in sorted(ROOT.rglob('results.csv')):
        run_dir = results_csv.parent
        with open(results_csv) as f:
            for row in csv.DictReader(f):
                try:
                    time_ms = int(row['time_ms'])
                except (KeyError, ValueError):
                    continue
                if time_ms >= TIMEOUT_MS:
                    continue
                out_path = run_dir / out_filename(row)
                if not out_path.exists():
                    suspicious_missing.append((out_path, row))
                    continue
                if out_path.stat().st_size == 0:
                    suspicious_empty.append((out_path, row))

    print(f'=== Empty .out files with sub-timeout time ({len(suspicious_empty)}) ===')
    for path, row in suspicious_empty[:30]:
        rel = path.relative_to(ROOT)
        print(f'  {rel}  time_ms={row["time_ms"]}  status={row["optimal"]}  obj={row.get("objective","-")}')
    if len(suspicious_empty) > 30:
        print(f'  ... and {len(suspicious_empty) - 30} more')

    print(f'\n=== Missing .out files (sub-timeout) ({len(suspicious_missing)}) ===')
    for path, row in suspicious_missing[:30]:
        rel = path.relative_to(ROOT)
        print(f'  {rel}  time_ms={row["time_ms"]}  status={row["optimal"]}')
    if len(suspicious_missing) > 30:
        print(f'  ... and {len(suspicious_missing) - 30} more')


if __name__ == '__main__':
    main()
