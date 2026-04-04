"""
For each solver, compute what percentage of contested instances it was wrong.

Rules:
  - UNSAT vs SOLVED: majority of unique solver names wins.
    If evenly split, both sides are wrong.
  - Optimal objective disagreement: majority objective value wins.
    If evenly split, all are wrong.

A solver's error rate = wrong_instances / contested_instances.
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

CSV_PATH = Path(__file__).parent.parent / "open-category-benchmarks" / "combined.csv"
TYPES_CSV_PATH = Path(__file__).parent / "problem_types.csv"

SOLVED_STATUSES = {"Satisfied", "Optimal", "AllSolutions"}


def load_problem_types(path: Path) -> dict[str, str]:
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[row["model"]] = row["type"]
    return types


def majority_value(values: list) -> object | None:
    """Return the majority value, or None if there is a tie at the top."""
    counts = Counter(values)
    top = counts.most_common(2)
    if len(top) == 1 or top[0][1] > top[1][1]:
        return top[0][0]
    return None  # tie


def main() -> None:
    problem_types = load_problem_types(TYPES_CSV_PATH)

    instances: dict[tuple, list[dict]] = defaultdict(list)
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            key = (row["year"], row["problem"], row["model"], row["name"])
            instances[key].append(row)

    # (solver, cores) -> {wrong: int, contested: int, total: int}
    Key = tuple[str, str]
    stats: dict[Key, dict[str, int]] = defaultdict(lambda: {"wrong": 0, "contested": 0, "total": 0})

    for group in instances.values():
        for key in {(r["solver"], r["cores"]) for r in group}:
            stats[key]["total"] += 1

    for (year, problem, model, name), group in instances.items():
        kind = problem_types.get(model)

        # ── UNSAT vs SOLVED ──────────────────────────────────────────────────
        unsat_rows  = [r for r in group if r["status"] in ("Unsatisfiable", "Unsat")]
        solved_rows = [r for r in group if r["status"] in SOLVED_STATUSES]

        if unsat_rows and solved_rows:
            # Majority vote by unique solver name (cores don't inflate the vote)
            unsat_names  = {r["solver"] for r in unsat_rows}
            solved_names = {r["solver"] for r in solved_rows}

            if len(solved_names) > len(unsat_names):
                wrong_names = unsat_names
            elif len(unsat_names) > len(solved_names):
                wrong_names = solved_names
            else:
                wrong_names = unsat_names | solved_names  # tie → all wrong

            contested_rows = unsat_rows + solved_rows
            seen: set[Key] = set()
            for r in contested_rows:
                k: Key = (r["solver"], r["cores"])
                if k in seen:
                    continue
                seen.add(k)
                stats[k]["contested"] += 1
                if r["solver"] in wrong_names:
                    stats[k]["wrong"] += 1

        # ── Optimal objective disagreements ──────────────────────────────────
        if kind in ("MIN", "MAX"):
            optimals = [r for r in group if r["status"] == "Optimal" and r["objective"]]
            if len(optimals) >= 2:
                # Best objective per solver (for majority vote)
                obj_by_solver: dict[str, float] = {}
                for r in optimals:
                    val = float(r["objective"])
                    existing = obj_by_solver.get(r["solver"])
                    if existing is None:
                        obj_by_solver[r["solver"]] = val
                    elif kind == "MIN":
                        obj_by_solver[r["solver"]] = min(existing, val)
                    else:
                        obj_by_solver[r["solver"]] = max(existing, val)

                if len(set(obj_by_solver.values())) > 1:
                    maj_obj = majority_value(list(obj_by_solver.values()))
                    # Track per (solver, cores) run
                    seen2: set[Key] = set()
                    for r in optimals:
                        k2: Key = (r["solver"], r["cores"])
                        if k2 in seen2:
                            continue
                        seen2.add(k2)
                        stats[k2]["contested"] += 1
                        solver_best = obj_by_solver[r["solver"]]
                        if maj_obj is None or solver_best != maj_obj:
                            stats[k2]["wrong"] += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"{'Solver':<40} {'Cores':>5}  {'Wrong':>6} / {'Contested':<10}  {'Contested%':>11}  {'Full%':>7}")
    print("─" * 90)
    rows = sorted(
        stats.items(),
        key=lambda x: (x[0][0], int(x[0][1])),
    )
    for (solver, cores), s in rows:
        contested_pct = s["wrong"] / s["contested"] * 100 if s["contested"] else 0.0
        full_pct      = s["wrong"] / s["total"]     * 100 if s["total"]     else 0.0
        print(f"{solver:<40} {cores:>5}  {s['wrong']:>6} / {s['contested']:<10}  {contested_pct:>10.1f}%  {full_pct:>6.2f}%")


if __name__ == "__main__":
    main()
