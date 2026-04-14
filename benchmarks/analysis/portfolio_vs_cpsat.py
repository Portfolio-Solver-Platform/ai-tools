"""
Head-to-head comparison of a portfolio against cp-sat(8c).

Per instance: who got better status, objective, time.
Summary: total time, solve rate, wins/ties/losses by year and problem type.

Usage:
    python portfolio_vs_cpsat.py
"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPEN_CSV = ROOT / "open-category-benchmarks" / "combined.csv"
PORTFOLIO_CSV = ROOT / "portfolios" / "combined.csv"
TYPES_CSV = ROOT / "open-category-benchmarks" / "problem_types.csv"
OUTPUT_CSV = Path(__file__).resolve().parent / "portfolio_vs_cpsat.csv"

SCHEDULE = "k1-8c-8s-v1"
SOLVER = "cp-sat"
CORES = 8

SOLVED = {"Optimal", "Satisfied", "Unsat"}
RANK = {"Optimal": 3, "Satisfied": 2, "Unsat": 2, "Unknown": 1, "Error": 0}


def load_problem_types(path):
    types = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            types[(row["problem"], row["model"])] = row["type"]
    return types


def compare(p, c, kind):
    """Return 'portfolio', 'cpsat', or 'tie'."""
    p_solved = p["status"] in SOLVED
    c_solved = c["status"] in SOLVED

    if p_solved and not c_solved:
        return "portfolio"
    if c_solved and not p_solved:
        return "cpsat"
    if not p_solved and not c_solved:
        return "tie"

    # Both solved
    p_rank = RANK[p["status"]]
    c_rank = RANK[c["status"]]

    if kind in ("MIN", "MAX"):
        p_obj = float(p["objective"]) if p["objective"] else None
        c_obj = float(c["objective"]) if c["objective"] else None

        # Optimal beats Satisfied
        if p_rank > c_rank:
            return "portfolio"
        if c_rank > p_rank:
            return "cpsat"

        # Same status — compare objectives
        if p_obj is not None and c_obj is not None and p_obj != c_obj:
            if kind == "MIN":
                return "portfolio" if p_obj < c_obj else "cpsat"
            else:
                return "portfolio" if p_obj > c_obj else "cpsat"

    # Same solve quality — compare time
    p_time = float(p["time_ms"])
    c_time = float(c["time_ms"])
    if p_time < c_time:
        return "portfolio"
    if c_time < p_time:
        return "cpsat"
    return "tie"


def main():
    problem_types = load_problem_types(TYPES_CSV)

    cpsat = {}
    with open(OPEN_CSV) as f:
        for r in csv.DictReader(f):
            if r["solver"] == SOLVER and int(r["cores"]) == CORES:
                cpsat[(r["problem"], r["name"])] = r

    portfolio = {}
    with open(PORTFOLIO_CSV) as f:
        for r in csv.DictReader(f):
            if r["schedule"] == SCHEDULE:
                portfolio[(r["problem"], r["name"])] = r

    instances = sorted(set(cpsat) & set(portfolio))

    rows = []
    for inst in instances:
        p = portfolio[inst]
        c = cpsat[inst]
        kind = problem_types.get((inst[0], p["model"]), "")
        winner = compare(p, c, kind)

        rows.append({
            "year": p["year"],
            "problem": inst[0],
            "name": inst[1],
            "type": kind,
            "winner": winner,
            "p_status": p["status"],
            "p_objective": p["objective"],
            "p_time_ms": p["time_ms"],
            "p_from": p["last_result_from"],
            "c_status": c["status"],
            "c_objective": c["objective"],
            "c_time_ms": c["time_ms"],
        })

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- Summary ---
    wins = {"portfolio": 0, "cpsat": 0, "tie": 0}
    p_solved = c_solved = 0
    p_time = c_time = 0.0
    by_year = defaultdict(lambda: {"portfolio": 0, "cpsat": 0, "tie": 0})
    by_type = defaultdict(lambda: {"portfolio": 0, "cpsat": 0, "tie": 0})

    for r in rows:
        wins[r["winner"]] += 1
        by_year[r["year"]][r["winner"]] += 1
        by_type[r["type"]][r["winner"]] += 1
        if r["p_status"] in SOLVED:
            p_solved += 1
        if r["c_status"] in SOLVED:
            c_solved += 1
        p_time += float(r["p_time_ms"])
        c_time += float(r["c_time_ms"])

    print(f"{SCHEDULE} vs {SOLVER}({CORES}c)")
    print(f"{'='*50}")
    print(f"Instances: {len(rows)}")
    print(f"Solved:    portfolio={p_solved}  cpsat={c_solved}")
    print(f"Total time: portfolio={p_time/1000:.0f}s  cpsat={c_time/1000:.0f}s")
    print(f"\nHead-to-head:")
    print(f"  Portfolio wins: {wins['portfolio']}")
    print(f"  CP-SAT wins:   {wins['cpsat']}")
    print(f"  Ties:          {wins['tie']}")

    print(f"\nBy year:")
    for y in sorted(by_year):
        d = by_year[y]
        print(f"  {y}: portfolio={d['portfolio']}  cpsat={d['cpsat']}  tie={d['tie']}")

    print(f"\nBy problem type:")
    for t in sorted(by_type):
        d = by_type[t]
        print(f"  {t}: portfolio={d['portfolio']}  cpsat={d['cpsat']}  tie={d['tie']}")

    print(f"\nPer-instance details written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
