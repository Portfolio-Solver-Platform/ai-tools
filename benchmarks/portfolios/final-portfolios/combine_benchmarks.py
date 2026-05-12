"""
Combine all per-portfolio per-year results.csv files under
portfolios-final/{portfolio}/{portfolio}-{year}/results.csv into a single combined.csv,
mirroring the open-category-benchmarks/combined.csv layout (minus open_category and wrong).
"""
import csv
from pathlib import Path

BASE = Path(__file__).parent
SOURCE_ROOT = BASE / "portfolios-final"
OUT_PATH = BASE / "combined.csv"

OUTPUT_FIELDS = ["solver", "cores", "year", "problem", "name", "model", "time_ms", "objective", "status"]

CORES = {
    "cpsat8":        8,
    "ek1-8c-8s-v2":  8,
    "k1-8c-8s-v1":   8,
}

STATUS_NORMALIZE = {
    "Unsatisfiable": "Unsat",
}


def main():
    all_rows = []
    for portfolio_dir in sorted(SOURCE_ROOT.iterdir()):
        if not portfolio_dir.is_dir():
            continue
        portfolio = portfolio_dir.name
        if portfolio not in CORES:
            print(f"WARN: unknown portfolio {portfolio}, skipping")
            continue
        for year_dir in sorted(portfolio_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            prefix = f"{portfolio}-"
            if not year_dir.name.startswith(prefix):
                continue
            year = year_dir.name[len(prefix):]
            results = year_dir / "results.csv"
            if not results.exists():
                continue
            with open(results, newline="") as f:
                for r in csv.DictReader(f):
                    status = STATUS_NORMALIZE.get(r["optimal"], r["optimal"])
                    all_rows.append({
                        "solver":    portfolio,
                        "cores":     CORES[portfolio],
                        "year":      year,
                        "problem":   r["problem"],
                        "name":      r["name"],
                        "model":     r["model"],
                        "time_ms":   r["time_ms"],
                        "objective": r["objective"],
                        "status":    status,
                    })

    print(f"Total: {len(all_rows)} rows")
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    main()
