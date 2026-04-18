"""
Combines all portfolio results.csv files into a single combined.csv.

Each portfolio folder contains either:
  - Year subfolders (e.g. handmade-foo/handmade-foo-2023/results.csv)
  - Direct results (e.g. k1-8c-8s-v1-2023/results.csv)

The year is extracted from the leaf folder name (last -YYYY suffix).

Usage:
    python combine_benchmarks.py all/        # all/portfolios/ → all/combined.csv
    python combine_benchmarks.py eligible/   # eligible/portfolios/ → eligible/combined.csv
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

OUTPUT_FIELDS = ["schedule", "year", "problem", "name", "model",
                 "time_ms", "objective", "status", "last_result_from"]


def extract_year(folder_name):
    m = re.search(r"-(\d{4})$", folder_name)
    return m.group(1) if m else None


EXPECTED_INSTANCES = 300


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="data directory (e.g. all/ or eligible/)")
    args = ap.parse_args()

    base = args.data_dir
    portfolios_dir = base / "portfolios"
    out_path = base / "combined.csv"

    by_schedule = defaultdict(list)

    for csv_path in sorted(portfolios_dir.rglob("results.csv")):
        year = extract_year(csv_path.parent.name)
        if year is None:
            print(f"WARNING: can't extract year from {csv_path.parent.name}, skipping")
            continue

        with open(csv_path) as f:
            for row in csv.DictReader(f):
                by_schedule[row["schedule"]].append({
                    "schedule": row["schedule"],
                    "year": year,
                    "problem": row["problem"],
                    "name": row["name"],
                    "model": row["model"],
                    "time_ms": row["time_ms"],
                    "objective": row["objective"],
                    "status": "Satisfied" if row["optimal"] == "Unknown" and row["objective"] else row["optimal"],
                    "last_result_from": row["last_result_from"],
                })

    all_rows = []
    for schedule, rows in sorted(by_schedule.items()):
        if len(rows) < EXPECTED_INSTANCES:
            print(f"SKIPPING {schedule}: {len(rows)}/{EXPECTED_INSTANCES} instances (incomplete)")
            continue
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["schedule"], int(r["year"]), r["problem"], r["name"]))

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    schedules = {r["schedule"] for r in all_rows}
    print(f"{len(all_rows)} rows from {len(schedules)} schedules written to {out_path}")


if __name__ == "__main__":
    main()
