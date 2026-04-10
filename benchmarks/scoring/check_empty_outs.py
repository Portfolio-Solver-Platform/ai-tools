"""
Finds entries where the .out file is empty but the solver did not hit the
20 minute (1,200,000 ms) timeout. An empty .out file combined with a short
runtime usually indicates a crashed / aborted run rather than a real timeout.
"""
import csv
from pathlib import Path

SOLVERS_DIR = Path(__file__).parent.parent / 'open-category-benchmarks' / 'solvers'
TIMEOUT_MS = 20 * 60 * 1000  # 20 minutes


def out_filename(row: dict) -> str:
    return (
        f"{row['problem']}-sep-{row['model']}-sep-{row['name']}"
        f"-sep-{row['solver']}-sep-{row['cores']}-sep-0.out"
    )


def main():
    suspicious: list[tuple[Path, dict]] = []

    for results_csv in sorted(SOLVERS_DIR.rglob('results.csv')):
        run_dir = results_csv.parent
        with open(results_csv) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or 'solver' not in reader.fieldnames:
                continue
            for row in reader:
                try:
                    time_ms = int(row['time_ms'])
                except (KeyError, ValueError):
                    continue
                if time_ms >= TIMEOUT_MS:
                    continue

                out_path = run_dir / out_filename(row)
                if not out_path.exists():
                    continue
                if out_path.stat().st_size == 0:
                    suspicious.append((out_path, row))

    print(f'=== Empty .out files with time_ms < {TIMEOUT_MS} ({len(suspicious)}) ===\n')
    for out_path, row in suspicious:
        rel = out_path.relative_to(SOLVERS_DIR)
        print(
            f'{rel}\n'
            f'  solver={row["solver"]} cores={row["cores"]} '
            f'time_ms={row["time_ms"]} status={row["status"]}'
        )


if __name__ == '__main__':
    main()
