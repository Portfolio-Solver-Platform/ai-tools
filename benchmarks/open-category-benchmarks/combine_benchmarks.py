"""
Combines ucloud solver results (tmp-solvers/combined.csv) with selected solvers
from allesio/mzn-challenge.csv into a single combined.csv.

Solvers taken from allesio: Picat, CPLEX, gecode, cp-sat
"""
import csv
from pathlib import Path

BASE = Path(__file__).parent
OUT_PATH = BASE / "combined.csv"
FIELDNAMES = ["solver", "cores", "year", "problem", "name", "model", "time_ms", "objective", "status"]

ALLESIO_SOLVERS = {"Picat", "CPLEX", "gecode", "cp-sat"}

STATUS_NORMALIZE = {
    "Unsatisfiable": "Unsat",
}


def read_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({field: row[field] for field in FIELDNAMES})
    return rows


def read_mzn_challenge(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["solver"] not in ALLESIO_SOLVERS:
                continue
            rows.append({field: row[field] for field in FIELDNAMES})
    return rows


def main():
    all_rows = []

    ucloud_csv = BASE / "tmp-solvers" / "combined.csv"
    if not ucloud_csv.exists():
        raise SystemExit(f"ERROR: {ucloud_csv} not found — run utils/get-ucloud-results.py first")
    rows = read_csv(ucloud_csv)
    print(f"tmp-solvers/combined.csv: {len(rows)} rows")
    all_rows.extend(rows)

    mzn_csv = BASE / "allesio" / "mzn-challenge.csv"
    if mzn_csv.exists():
        rows = read_mzn_challenge(mzn_csv)
        print(f"allesio/mzn-challenge.csv: {len(rows)} rows (solvers: {', '.join(sorted(ALLESIO_SOLVERS))})")
        all_rows.extend(rows)

    for r in all_rows:
        r["status"] = STATUS_NORMALIZE.get(r["status"], r["status"])

    print(f"\nTotal: {len(all_rows)} rows")

    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
