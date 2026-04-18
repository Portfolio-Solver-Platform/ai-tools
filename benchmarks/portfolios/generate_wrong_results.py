"""
Detects provably wrong portfolio results by cross-referencing against
trusted solver results from the open-category benchmarks.

Trusted solvers: gecode, chuffed, cp-sat.

A portfolio result is wrong if:
  - It claims Unsat but trusted solvers found solutions (false_unsat)
  - It claims Optimal with an objective worse than a trusted solver's
    feasible solution (wrong_optimal)

Usage:
    python generate_wrong_results.py all/
    python generate_wrong_results.py eligible/
"""
import argparse
import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
OPEN_CSV = ROOT / "open-category-benchmarks" / "combined.csv"
TYPES_CSV = ROOT / "open-category-benchmarks" / "problem_types.csv"

TRUSTED_SOLVERS = {"org.gecode.gecode", "org.chuffed.chuffed", "cp-sat"}
SOLVED_STATUSES = {"Satisfied", "Optimal"}


def load_problem_types(path):
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row["problem"], row["model"])] = row["type"]
    return types


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path,
                    help="data directory (e.g. all/ or eligible/)")
    args = ap.parse_args()

    portfolio_csv = args.data_dir / "combined.csv"
    output_csv = args.data_dir / "wrong_results.csv"

    problem_types = load_problem_types(TYPES_CSV)

    # Build ground truth from open-category trusted solvers
    trusted_solved = set()          # (year, problem, name) where trusted found a solution
    trusted_opt_obj = {}            # (year, problem, name) -> float, if trusted proved Optimal
    trusted_best_feasible = {}      # (year, problem, name) -> float, best feasible from trusted

    with open(OPEN_CSV) as f:
        for r in csv.DictReader(f):
            if r["solver"] not in TRUSTED_SOLVERS:
                continue
            key = (r["year"], r["problem"], r["name"])
            kind = problem_types.get((r["problem"], r["model"]))

            if r["status"] in SOLVED_STATUSES:
                trusted_solved.add(key)

            if kind in ("MIN", "MAX") and r["objective"]:
                val = float(r["objective"])
                # Track best feasible
                prev = trusted_best_feasible.get(key)
                if prev is None:
                    trusted_best_feasible[key] = val
                elif kind == "MIN":
                    trusted_best_feasible[key] = min(prev, val)
                else:
                    trusted_best_feasible[key] = max(prev, val)

                # Track proven optimal
                if r["status"] == "Optimal":
                    if key in trusted_opt_obj:
                        existing = trusted_opt_obj[key]
                        if existing != val:
                            print(f"WARNING: trusted solvers disagree on {key}: {existing} vs {val}")
                    else:
                        trusted_opt_obj[key] = val

    # Check portfolio results against ground truth
    wrong_rows = []

    with open(portfolio_csv) as f:
        for r in csv.DictReader(f):
            key = (r["year"], r["problem"], r["name"])
            kind = problem_types.get((r["problem"], r["model"]))

            # False UNSAT: portfolio says Unsat but trusted found solutions
            if r["status"] == "Unsat" and key in trusted_solved:
                wrong_rows.append({
                    "schedule": r["schedule"],
                    "year": r["year"],
                    "problem": r["problem"],
                    "name": r["name"],
                    "reason": "false_unsat",
                    "last_result_from": r["last_result_from"],
                })
                continue

            # Wrong Optimal: portfolio claims Optimal with worse objective
            if r["status"] == "Optimal" and kind in ("MIN", "MAX") and r["objective"]:
                val = float(r["objective"])

                # Check against trusted proven optimal
                if key in trusted_opt_obj and val != trusted_opt_obj[key]:
                    wrong_rows.append({
                        "schedule": r["schedule"],
                        "year": r["year"],
                        "problem": r["problem"],
                        "name": r["name"],
                        "reason": "wrong_optimal",
                        "last_result_from": r["last_result_from"],
                    })
                    continue

                # Check against trusted best feasible
                if key in trusted_best_feasible:
                    feasible = trusted_best_feasible[key]
                    is_worse = (kind == "MIN" and val > feasible) or \
                               (kind == "MAX" and val < feasible)
                    if is_worse:
                        wrong_rows.append({
                            "schedule": r["schedule"],
                            "year": r["year"],
                            "problem": r["problem"],
                            "name": r["name"],
                            "reason": "wrong_optimal",
                            "last_result_from": r["last_result_from"],
                        })

    wrong_rows.sort(key=lambda r: (r["schedule"], r["year"], r["problem"], r["name"]))

    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["schedule", "year", "problem", "name", "reason", "last_result_from"])
        w.writeheader()
        w.writerows(wrong_rows)

    reason_counts = Counter(r["reason"] for r in wrong_rows)
    print(f"Wrote {len(wrong_rows)} wrong results to {output_csv}")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
