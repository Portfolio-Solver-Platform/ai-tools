#!/usr/bin/env python3
"""
Collect results.csv files from selected solvers into a single tmp-solvers/ directory,
then combine them into a single combined.csv with a year column inserted after cores.

Usage: python utils/get-ucloud-results.py [--solvers-dir PATH] [--output-dir PATH]
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

SOLVERS_OF_INTEREST = {
    "choco", "chuffed", "coin-bc", "cplex", "cpsat", "dexter", "gecode", "highs", "izplus",
    "gurobi", "huub", "pumpkin", "picat", "scip", "yuck",
}


def collect_results(solvers_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    collected = []

    for solver_dir in sorted(solvers_dir.iterdir()):
        if not solver_dir.is_dir() or solver_dir.name not in SOLVERS_OF_INTEREST:
            continue

        for run_dir in sorted(solver_dir.rglob("*")):
            if not run_dir.is_dir():
                continue
            results_csv = run_dir / "results.csv"
            if not results_csv.exists():
                continue

            dest = output_dir / f"{run_dir.name}.csv"
            if dest.exists():
                print(f"  WARNING: {dest.name} already exists, overwriting")
            shutil.copy2(results_csv, dest)
            print(f"  {solver_dir.name}/{run_dir.name}/results.csv -> {dest.name}")
            collected.append(dest)

    return collected


def combine_results(csv_files: list[Path], output_path: Path) -> None:
    year_re = re.compile(r"-(\d{4})$")

    with output_path.open("w", newline="") as out_f:
        writer = None

        for csv_path in sorted(csv_files):
            stem = csv_path.stem  # e.g. "choco1-2025"
            m = year_re.search(stem)
            year = m.group(1) if m else "unknown"

            with csv_path.open(newline="") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    # Insert year after cores
                    new_row = {}
                    for col in reader.fieldnames:
                        new_row[col] = row[col]
                        if col == "cores":
                            new_row["year"] = year

                    if writer is None:
                        writer = csv.DictWriter(out_f, fieldnames=list(new_row.keys()))
                        writer.writeheader()
                    writer.writerow(new_row)

    print(f"  Combined {len(csv_files)} files -> {output_path}")


def main() -> None:
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solvers-dir",
        type=Path,
        default=repo_root / "solvers",
        help="Path to the solvers directory (default: <repo-root>/solvers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "tmp-solvers",
        help="Destination directory for collected CSVs (default: <repo-root>/tmp-solvers)",
    )
    args = parser.parse_args()

    if not args.solvers_dir.exists():
        raise SystemExit(f"ERROR: solvers directory not found: {args.solvers_dir}")

    combined_path = args.output_dir / "combined.csv"

    print(f"Collecting from: {args.solvers_dir}")
    print(f"Output to:       {args.output_dir}")
    print(f"Combined CSV:    {combined_path}\n")

    collected = collect_results(args.solvers_dir, args.output_dir)
    print(f"\nCombining {len(collected)} files...")
    combine_results(collected, combined_path)
    print(f"\nDone: {len(collected)} files collected.")


if __name__ == "__main__":
    main()
