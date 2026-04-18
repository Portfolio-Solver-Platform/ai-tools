"""
Adds a 'wrong' column to combined.csv by joining against wrong_results.csv.

Usage:
    python add_wrong_column.py all/
    python add_wrong_column.py eligible/
"""
import argparse
import csv
from pathlib import Path

JOIN_KEYS = ("schedule", "year", "problem", "name")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="data directory (e.g. all/ or eligible/)")
    args = ap.parse_args()

    combined_csv = args.data_dir / "combined.csv"
    wrong_csv = args.data_dir / "wrong_results.csv"

    wrong = set()
    with open(wrong_csv) as f:
        for row in csv.DictReader(f):
            wrong.add(tuple(row[k] for k in JOIN_KEYS))

    rows = []
    with open(combined_csv) as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in JOIN_KEYS)
            row["wrong"] = key in wrong
            rows.append(row)

    with open(combined_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_wrong = sum(1 for r in rows if r["wrong"])
    print(f"Marked {n_wrong}/{len(rows)} rows as wrong")


if __name__ == "__main__":
    main()
