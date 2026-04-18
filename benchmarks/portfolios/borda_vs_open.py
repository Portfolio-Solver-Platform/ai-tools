"""
Computes Borda scores for portfolio schedules against the open-category solvers.

Combines portfolio results with open-category results and scores all configs
against the 15 open-category opponents, same as best-k-portfolios.py does
for individual solvers.

Usage:
    python borda_vs_open.py all/
    python borda_vs_open.py eligible/
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores, load_problem_types

ROOT = Path(__file__).resolve().parent.parent.parent
OPEN_CSV = ROOT / "benchmarks" / "open-category-benchmarks" / "combined.csv"
TYPES_CSV = ROOT / "benchmarks" / "open-category-benchmarks" / "problem_types.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="data directory (e.g. all/ or eligible/)")
    args = ap.parse_args()

    portfolio_csv = args.data_dir / "combined.csv"

    problem_types = load_problem_types(TYPES_CSV)

    # Load open-category rows
    with open(OPEN_CSV) as f:
        open_rows = list(csv.DictReader(f))

    open_category = {(r["solver"], int(r["cores"])) for r in open_rows if r["open_category"] == "True"}

    # Load portfolio rows, renaming schedule -> solver
    with open(portfolio_csv) as f:
        for r in csv.DictReader(f):
            open_rows.append({
                "solver": r["schedule"],
                "cores": 8,
                "problem": r["problem"],
                "name": r["name"],
                "model": r["model"],
                "time_ms": r["time_ms"],
                "objective": r["objective"],
                "status": r["status"],
                "wrong": r["wrong"],
            })

    scores, configs, instances = borda_scores(open_rows, problem_types, opponents=open_category)

    # Separate portfolio configs from solver configs
    portfolio_schedules = set()
    with open(portfolio_csv) as f:
        for r in csv.DictReader(f):
            portfolio_schedules.add(r["schedule"])

    # Per-instance year mapping for per-year breakdown
    instance_year = {}
    with open(portfolio_csv) as f:
        for r in csv.DictReader(f):
            instance_year[(r["problem"], r["name"])] = r["year"]

    years = sorted(set(instance_year.values()))

    print(f"{'Schedule':<35} {'Total':>8}", end="")
    for y in years:
        print(f"  {y:>8}", end="")
    print()
    print("-" * (35 + 9 + len(years) * 10))

    results = []
    for i, cfg in enumerate(configs):
        is_portfolio = cfg[0] in portfolio_schedules
        is_open = cfg in open_category
        if not is_portfolio and not is_open:
            continue
        total = scores[i].sum()
        per_year = {}
        for j, inst in enumerate(instances):
            y = instance_year.get(inst)
            if y:
                per_year[y] = per_year.get(y, 0) + scores[i, j]
        label = cfg[0] if is_portfolio else f"{cfg[0]}({cfg[1]}c)"
        kind = "portfolio" if is_portfolio else "solver"
        results.append((total, label, per_year, kind))

    results.sort(reverse=True)

    for total, name, per_year, kind in results:
        marker = " " if kind == "portfolio" else "*"
        print(f"{marker} {name:<34} {total:>8.1f}", end="")
        for y in years:
            print(f"  {per_year.get(y, 0):>8.1f}", end="")
        print()
    print()
    print("* = open-category solver")


if __name__ == "__main__":
    main()
