"""
Measure portfolio overhead: theoretical (VBS of individual solvers at 1 core)
vs actual (measured portfolio on 8 shared cores).

Theoretical = no interference between solvers (each performs as in isolation).
Compares by total time and Borda score against open-category opponents.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.borda import borda_scores

DATA_PATH = Path(__file__).resolve().parent / "solver_portfolio_results.json"

OPEN_CATEGORY = {
    "cp-sat": 8, "choco": 8, "gecode": 8, "CPLEX": 8,
    "chuffed": 1, "Huub": 1, "Picat": 1,
}
EXCLUDE = {"portfolio", "portfolio2", "portfolio3", "static-scheduler"}


def map_status(entry):
    if entry["has_solution"] and entry["optimal"] == "Optimal":
        return "Satisfied" if entry["search"] == "Satisfy" else "Optimal"
    if entry["has_solution"]:
        return "Satisfied"
    if entry["optimal"] == "Optimal":
        return "Unsat"
    return "Unknown"


def pick_best(results, kind):
    """Pick the result that would score highest in Borda comparison."""
    def key(r):
        status = map_status(r)
        solved = status in ("Optimal", "Satisfied", "Unsat")
        complete = status in ("Optimal", "Unsat")
        obj = r["objective"]
        if kind == "SAT" or obj is None:
            obj_val = 0
        elif kind == "MIN":
            obj_val = -obj
        else:
            obj_val = obj
        return (solved, obj_val, complete, -r["time"])
    return max(results, key=key)


def build_rows(data):
    """Convert JSON to borda-compatible rows + problem_types dict."""
    rows = []
    problem_types = {}
    kind_map = {"Satisfy": "SAT", "Minimise": "MIN", "Maximise": "MAX"}

    # Map lowercase solver names in combos to actual JSON keys
    key_lookup = {k.lower(): k for k in data}

    def add(solver, cores, entry):
        rows.append({
            "solver": solver, "cores": cores,
            "problem": entry["model"], "name": entry["name"],
            "model": entry["model"],
            "status": map_status(entry),
            "time_ms": entry["time"],
            "objective": entry["objective"],
            "wrong": False,
        })
        problem_types[(entry["model"], entry["model"])] = kind_map[entry["search"]]

    # Open-category opponents
    for solver, cores in OPEN_CATEGORY.items():
        for entry in data[solver][str(cores)]:
            add(solver, cores, entry)

    # Portfolio combos
    combos = sorted(
        [k for k in data if "," in k and k not in EXCLUDE],
        key=lambda c: (len(c.split(",")), c),
    )

    for combo in combos:
        # Actual measured results
        for entry in data[combo]["8"]:
            add(combo, 8, entry)

        # Theoretical VBS (best of constituent solvers at 1 core)
        solvers = combo.split(",")
        by_instance = {}
        for solver in solvers:
            actual_key = key_lookup[solver.lower()]
            for entry in data[actual_key]["1"]:
                key = (entry["model"], entry["name"])
                by_instance.setdefault(key, []).append(entry)

        theo_name = f"theo:{combo}"
        for (model, name), results in by_instance.items():
            kind = problem_types.get((model, model), "SAT")
            best = pick_best(results, kind)
            add(theo_name, 8, best)

    return rows, problem_types, combos


def main():
    with open(DATA_PATH) as f:
        data = json.load(f)

    rows, problem_types, combos = build_rows(data)
    scores, configs, instances = borda_scores(rows, problem_types)
    config_idx = {c: i for i, c in enumerate(configs)}

    # Compute theoretical min-time per combo (min across solvers at 1c per instance)
    key_lookup = {k.lower(): k for k in data}
    theo_min_time = {}
    actual_time = {}
    for combo in combos:
        solvers = combo.split(",")
        by_inst = {}
        for s in solvers:
            for e in data[key_lookup[s.lower()]]["1"]:
                k = (e["model"], e["name"])
                by_inst.setdefault(k, []).append(e["time"])
        theo_min_time[combo] = sum(min(times) for times in by_inst.values())
        actual_time[combo] = sum(e["time"] for e in data[combo]["8"])

    hdr = (f"{'Portfolio':<55} {'#':>2}  {'Actual':>7} {'Theo':>7} {'Δ Borda':>8}  "
           f"{'Actual(s)':>10} {'Theo(s)':>10} {'Overhead':>8}")
    print(hdr)
    print("=" * len(hdr))

    by_size = {}
    for combo in combos:
        n = len(combo.split(","))
        actual_cfg = (combo, 8)
        theo_cfg = (f"theo:{combo}", 8)

        ab = scores[config_idx[actual_cfg]].sum()
        tb = scores[config_idx[theo_cfg]].sum()

        at = actual_time[combo] / 1000
        tt = theo_min_time[combo] / 1000

        pct = ((at - tt) / tt * 100) if tt > 0 else 0
        print(f"{combo:<55} {n:>2}  {ab:>7.1f} {tb:>7.1f} {ab - tb:>+8.1f}  "
              f"{at:>10.0f} {tt:>10.0f} {pct:>+7.1f}%")

        by_size.setdefault(n, []).append((ab, tb, at, tt))

    print()
    print(f"{'Size':>4}  {'Avg Borda %':>11}  {'Avg Time %':>10}")
    print("-" * 32)
    for n in sorted(by_size):
        entries = by_size[n]
        borda_pcts = [(ab - tb) / tb * 100 for ab, tb, _, _ in entries]
        time_pcts = [(at - tt) / tt * 100 for _, _, at, tt in entries]
        print(f"{n:>4}  {sum(borda_pcts)/len(borda_pcts):>+10.1f}%  {sum(time_pcts)/len(time_pcts):>+9.1f}%")


if __name__ == "__main__":
    main()
