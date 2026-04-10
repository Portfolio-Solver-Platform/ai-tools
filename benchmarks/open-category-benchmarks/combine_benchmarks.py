"""
Combines ucloud solver results (tmp-solvers/combined.csv) with selected solvers
from allesio/mzn-challenge.csv into a single combined.csv.

Solvers taken from allesio: Picat, CPLEX, gecode, cp-sat
"""
import csv
from pathlib import Path

BASE = Path(__file__).parent
OUT_PATH = BASE / "combined.csv"
INPUT_FIELDS = ["solver", "cores", "year", "problem", "name", "model", "time_ms", "objective", "status"]
# open_category: True for the row that represents each solver's best variant per year
# (the one with the most cores). These are the rows that compete in the "open" category.
OUTPUT_FIELDS = INPUT_FIELDS + ["open_category"]

ALLESIO_SOLVERS = {"Picat", "CPLEX", "gecode", "cp-sat"}

STATUS_NORMALIZE = {
    "Unsatisfiable": "Unsat",
}


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({field: row[field] for field in INPUT_FIELDS})
    return rows


def annotate_open_category(rows: list[dict]) -> None:
    """Mark each solver's highest-core variant (per year) as the open-category rep."""
    max_cores: dict[tuple, int] = {}
    for r in rows:
        key = (r["solver"], r["year"])
        c = int(r["cores"])
        if c > max_cores.get(key, -1):
            max_cores[key] = c
    for r in rows:
        key = (r["solver"], r["year"])
        r["open_category"] = "True" if int(r["cores"]) == max_cores[key] else "False"


def main():
    all_rows = []

    ucloud_csv = BASE / "tmp-solvers" / "combined.csv"
    if not ucloud_csv.exists():
        raise SystemExit(f"ERROR: {ucloud_csv} not found — run utils/get-ucloud-results.py first")
    rows = read_csv(ucloud_csv)
    print(f"tmp-solvers/combined.csv: {len(rows)} rows")
    all_rows.extend(rows)

    for r in all_rows:
        r["status"] = STATUS_NORMALIZE.get(r["status"], r["status"])

    annotate_open_category(all_rows)

    print(f"\nTotal: {len(all_rows)} rows")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
