"""
Trust-no-one validation of solver and portfolio results.

Combines open-category and portfolio data, deduplicates by underlying solver
identity, then uses majority vote and contradiction detection to flag
provably wrong results.

Checks:
  1. False Unsat: majority found solutions, but some claim Unsat
  2. Wrong Optimal: majority of Optimal claims agree on objective X,
     minority claims Y
  3. Optimal worse than feasible: a solver found a feasible objective
     better than another solver's claimed Optimal

Usage:
    python validate_results.py
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPEN_CSV = ROOT / "open-category-benchmarks" / "combined.csv"
PORTFOLIO_CSV = ROOT / "portfolios" / "combined.csv"
TYPES_CSV = ROOT / "open-category-benchmarks" / "problem_types.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "flagged_results.csv"

SOLVED = {"Optimal", "Satisfied"}


def load_problem_types(path):
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row["problem"], row["model"])] = row["type"]
    return types


def main():
    problem_types = load_problem_types(TYPES_CSV)

    # Collect per-instance results, keyed by (year, problem, name)
    # Each entry: {solver_name: [{status, objective, source, source_id}]}
    instances = defaultdict(lambda: defaultdict(list))
    instance_model = {}

    with open(OPEN_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["year"], r["problem"], r["name"])
            instance_model[key] = r["model"]
            instances[key][r["solver"]].append({
                "status": r["status"],
                "objective": r["objective"],
                "source": "solver",
                "source_id": f"{r['solver']}({r['cores']}c)",
            })

    with open(PORTFOLIO_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["year"], r["problem"], r["name"])
            instance_model[key] = r["model"]
            solver = r["last_result_from"]
            if not solver:
                continue
            instances[key][solver].append({
                "status": r["status"],
                "objective": r["objective"],
                "source": "portfolio",
                "source_id": r["schedule"],
            })

    flagged = []

    for (year, problem, name), by_solver in sorted(instances.items()):
        model = instance_model.get((year, problem, name), "")
        kind = problem_types.get((problem, model))

        # Best result per unique solver: pick best status, then best objective
        solver_best = {}
        for solver_name, entries in by_solver.items():
            best_status = None
            best_obj = None
            sources = set()
            for e in entries:
                sources.add(e["source_id"])
                # Prefer Optimal > Satisfied > Unsat > Unknown > Error
                rank = {"Optimal": 4, "Satisfied": 3, "Unsat": 2, "Unknown": 1, "Error": 0}
                r = rank.get(e["status"], -1)
                if best_status is None or r > rank.get(best_status, -1):
                    best_status = e["status"]
                if e["objective"]:
                    val = float(e["objective"])
                    if best_obj is None:
                        best_obj = val
                    elif kind == "MIN":
                        best_obj = min(best_obj, val)
                    elif kind == "MAX":
                        best_obj = max(best_obj, val)
            solver_best[solver_name] = {
                "status": best_status,
                "objective": best_obj,
                "sources": sources,
            }

        # --- Check 1: False Unsat ---
        solved_solvers = {s for s, b in solver_best.items() if b["status"] in SOLVED}
        unsat_solvers = {s for s, b in solver_best.items() if b["status"] == "Unsat"}

        if len(solved_solvers) >= 2 and unsat_solvers:
            for s in unsat_solvers:
                flagged.append({
                    "year": year,
                    "problem": problem,
                    "name": name,
                    "solver": s,
                    "sources": "; ".join(sorted(solver_best[s]["sources"])),
                    "reason": "false_unsat",
                    "detail": f"{len(solved_solvers)} solvers found solutions",
                })

        if kind not in ("MIN", "MAX"):
            continue

        # --- Check 2: Wrong Optimal (majority vote) ---
        optimal_solvers = {
            s: b["objective"] for s, b in solver_best.items()
            if b["status"] == "Optimal" and b["objective"] is not None
        }

        if len(optimal_solvers) >= 2:
            obj_votes = Counter(optimal_solvers.values())
            if len(obj_votes) > 1:
                majority_obj, majority_count = obj_votes.most_common(1)[0]
                second_count = obj_votes.most_common(2)[1][1]
                if majority_count > second_count:
                    for s, obj in optimal_solvers.items():
                        if obj != majority_obj:
                            flagged.append({
                                "year": year,
                                "problem": problem,
                                "name": name,
                                "solver": s,
                                "sources": "; ".join(sorted(solver_best[s]["sources"])),
                                "reason": "wrong_optimal",
                                "detail": f"claims optimal={obj}, majority says {majority_obj} ({majority_count} vs {second_count})",
                            })
                else:
                    # Tie — flag for manual review
                    solvers_per_obj = defaultdict(list)
                    for s, obj in optimal_solvers.items():
                        solvers_per_obj[obj].append(s)
                    detail = "; ".join(f"obj={o}: {', '.join(ss)}" for o, ss in sorted(solvers_per_obj.items()))
                    flagged.append({
                        "year": year,
                        "problem": problem,
                        "name": name,
                        "solver": "TIE",
                        "sources": "",
                        "reason": "optimal_tie",
                        "detail": detail,
                    })

        # --- Check 3: Optimal worse than any feasible ---
        best_feasible = None
        best_feasible_solver = None
        for s, b in solver_best.items():
            if b["objective"] is not None:
                val = b["objective"]
                if best_feasible is None:
                    best_feasible = val
                    best_feasible_solver = s
                elif kind == "MIN" and val < best_feasible:
                    best_feasible = val
                    best_feasible_solver = s
                elif kind == "MAX" and val > best_feasible:
                    best_feasible = val
                    best_feasible_solver = s

        if best_feasible is not None:
            for s, obj in optimal_solvers.items():
                is_worse = (kind == "MIN" and obj > best_feasible) or \
                           (kind == "MAX" and obj < best_feasible)
                if is_worse:
                    # Avoid duplicate if already flagged as wrong_optimal
                    already = any(
                        f["year"] == year and f["problem"] == problem
                        and f["name"] == name and f["solver"] == s
                        and f["reason"] == "wrong_optimal"
                        for f in flagged
                    )
                    if not already:
                        flagged.append({
                            "year": year,
                            "problem": problem,
                            "name": name,
                            "solver": s,
                            "sources": "; ".join(sorted(solver_best[s]["sources"])),
                            "reason": "optimal_worse_than_feasible",
                            "detail": f"claims optimal={obj}, but {best_feasible_solver} found {best_feasible}",
                        })

    flagged.sort(key=lambda r: (r["reason"], r["year"], r["problem"], r["name"], r["solver"]))

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["year", "problem", "name", "solver", "sources", "reason", "detail"])
        w.writeheader()
        w.writerows(flagged)

    reason_counts = Counter(r["reason"] for r in flagged)
    print(f"Wrote {len(flagged)} flagged results to {OUTPUT_CSV}")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
