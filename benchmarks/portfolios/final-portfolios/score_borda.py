"""
Compute per-instance Borda scores for each portfolio under three different comparisons:
  - cpsat8_k1_ek1  (3-way: cpsat8 vs k1 vs ek1)
  - cpsat8_k1      (2-way: cpsat8 vs k1)
  - cpsat8_ek1     (2-way: cpsat8 vs ek1)

Long-format output: one row per (instance, solver, comparison).
Skips a comparison on an instance if any required portfolio's row is missing for that instance.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
ROOT = BASE.parent.parent.parent
COMBINED_CSV = BASE / "combined.csv"
TYPES_CSV    = ROOT / "benchmarks" / "open-category-benchmarks" / "problem_types.csv"
OUT_CSV      = BASE / "borda_per_instance.csv"

sys.path.insert(0, str(ROOT / "benchmarks" / "scoring"))
from borda import pairwise_score, load_problem_types

CPSAT = "cpsat8"
K1    = "k1-8c-8s-v1"
EK1   = "ek1-8c-8s-v2"

COMPARISONS = {
    "cpsat8_k1_ek1": (CPSAT, K1, EK1),
    "cpsat8_k1":     (CPSAT, K1),
    "cpsat8_ek1":    (CPSAT, EK1),
}

OUTPUT_FIELDS = ["year", "problem", "model", "name", "solver", "comparison", "borda"]


def main():
    problem_types = load_problem_types(TYPES_CSV)

    # Group by instance key
    instances: dict[tuple, dict[str, dict]] = defaultdict(dict)
    with open(COMBINED_CSV, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["year"], r["problem"], r["model"], r["name"])
            instances[key][r["solver"]] = r

    out_rows = []
    skipped_unknown_type = 0
    skipped_missing = defaultdict(int)

    for (year, problem, model, name), by_solver in instances.items():
        kind = problem_types.get((problem, model))
        if kind is None:
            skipped_unknown_type += 1
            continue

        for comp_name, solvers_in_comp in COMPARISONS.items():
            rows = [by_solver.get(s) for s in solvers_in_comp]
            if any(r is None for r in rows):
                skipped_missing[comp_name] += 1
                continue

            for i, s in enumerate(rows):
                borda = 0.0
                for j, s2 in enumerate(rows):
                    if i == j:
                        continue
                    borda += pairwise_score(s, s2, kind)
                out_rows.append({
                    "year":       year,
                    "problem":    problem,
                    "model":      model,
                    "name":       name,
                    "solver":     solvers_in_comp[i],
                    "comparison": comp_name,
                    "borda":      f"{borda:.6f}",
                })

    print(f"Total instances scanned: {len(instances)}")
    if skipped_unknown_type:
        print(f"Skipped (unknown problem/model type): {skipped_unknown_type}")
    for comp, n in skipped_missing.items():
        print(f"Skipped in {comp} (missing portfolio row): {n}")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
