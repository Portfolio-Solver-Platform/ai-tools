"""
Adds a 'wrong' column to combined.csv by joining against wrong_results.csv.

Usage:
    python add_wrong_column.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMBINED_CSV = ROOT / "combined.csv"
WRONG_CSV = ROOT / "wrong_results.csv"

JOIN_KEYS = ("solver", "cores", "year", "problem", "name")


def main():
    wrong = set()
    with open(WRONG_CSV) as f:
        for row in csv.DictReader(f):
            wrong.add(tuple(row[k] for k in JOIN_KEYS))

    rows = []
    with open(COMBINED_CSV) as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in JOIN_KEYS)
            row["wrong"] = key in wrong
            rows.append(row)

    with open(COMBINED_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_wrong = sum(1 for r in rows if r["wrong"])
    print(f"Marked {n_wrong}/{len(rows)} rows as wrong")


if __name__ == "__main__":
    main()
