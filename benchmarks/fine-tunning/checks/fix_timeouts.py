#!/usr/bin/env python3
"""
One-shot fix for fine-tuning benchmark CSVs that recorded incorrect time_ms.

Adapted from parasol/checks/fix_timeouts.py — points at fine-tunning/k1-8c-8s-v1/.

Walks k1-8c-8s-v1/, looks up each row's problem type from
scoring/problem_types.csv, and rewrites time_ms = 1200000 (the 20-min timeout)
for any row whose `optimal` column is not in {Optimal, Unsat}. The original CSV
is copied to results.csv.bak before being overwritten.

Fails loudly if any model in the data is missing from problem_types.csv.
"""

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "k1-8c-8s-v1"
TYPES_CSV = ROOT.parent / "open-category-benchmarks" / "problem_types.csv"

TIMEOUT_MS = 1_200_000
SOLVED_STATUSES = {"Optimal", "Unsat"}


def load_problem_types() -> dict[str, str]:
    types: dict[str, str] = {}
    with open(TYPES_CSV) as f:
        for r in csv.DictReader(f):
            types[r["model"]] = r["type"]
    return types


def fix_csv(csv_path: Path, problem_types: dict[str, str]) -> tuple[int, int, set[str]]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    missing: set[str] = set()
    fixed = 0
    for r in rows:
        if r["model"] not in problem_types:
            missing.add(r["model"])
            continue
        if r["optimal"] not in SOLVED_STATUSES:
            if r["time_ms"] != str(TIMEOUT_MS):
                r["time_ms"] = str(TIMEOUT_MS)
                fixed += 1

    if fixed > 0:
        backup = csv_path.with_suffix(".csv.bak")
        shutil.copy2(csv_path, backup)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows), fixed, missing


def main():
    problem_types = load_problem_types()
    if not problem_types:
        print(f"ERROR: no problem types loaded from {TYPES_CSV}", file=sys.stderr)
        sys.exit(1)

    csv_paths = sorted(DATA_ROOT.rglob("results.csv"))
    if not csv_paths:
        print(f"No results.csv files found under {DATA_ROOT}", file=sys.stderr)
        sys.exit(1)

    all_missing: set[str] = set()
    per_config: list[tuple[Path, int, int]] = []

    for csv_path in csv_paths:
        rows, fixed, missing = fix_csv(csv_path, problem_types)
        all_missing |= missing
        per_config.append((csv_path, rows, fixed))

    print(f"{'config':<70s}  {'rows':>6s}  {'fixed':>6s}")
    print("-" * 88)
    total_rows = total_fixed = 0
    for csv_path, rows, fixed in per_config:
        rel = csv_path.relative_to(DATA_ROOT).parent
        marker = "  ←" if fixed else ""
        print(f"{str(rel):<70s}  {rows:>6d}  {fixed:>6d}{marker}")
        total_rows += rows
        total_fixed += fixed
    print("-" * 88)
    print(f"{'TOTAL':<70s}  {total_rows:>6d}  {total_fixed:>6d}")

    if all_missing:
        print(f"\nERROR: {len(all_missing)} models not found in {TYPES_CSV}:", file=sys.stderr)
        for m in sorted(all_missing):
            print(f"  - {m}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
