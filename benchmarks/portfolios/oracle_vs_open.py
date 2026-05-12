"""
Theoretical-best (oracle) Borda scores against the 15 open-category solvers.

Two oracle pools:
  1. Best portfolio per instance      — upper bound for portfolio selection
                                        from this combined.csv.
  2. Best portfolio or open-cat solver — absolute ceiling against the
                                        15 open-category opponents.

Reports total, per-year, and max possible (15 * n_instances).

Usage:
    python oracle_vs_open.py all/
    python oracle_vs_open.py eligible/
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
    ap.add_argument("data_dir", type=Path)
    args = ap.parse_args()

    portfolio_csv = args.data_dir / "combined.csv"
    problem_types = load_problem_types(TYPES_CSV)

    with open(OPEN_CSV) as f:
        open_rows = list(csv.DictReader(f))
    open_category = {(r["solver"], int(r["cores"]))
                     for r in open_rows if r["open_category"] == "True"}

    portfolio_schedules = set()
    instance_year = {}
    with open(portfolio_csv) as f:
        for r in csv.DictReader(f):
            portfolio_schedules.add(r["schedule"])
            instance_year[(r["problem"], r["name"])] = r["year"]
            open_rows.append({
                "solver": r["schedule"], "cores": 8,
                "problem": r["problem"], "name": r["name"], "model": r["model"],
                "time_ms": r["time_ms"], "objective": r["objective"],
                "status": r["status"], "wrong": r["wrong"],
            })

    scores, configs, instances = borda_scores(open_rows, problem_types,
                                              opponents=open_category)

    portfolio_idxs = [i for i, c in enumerate(configs) if c[0] in portfolio_schedules]
    open_idxs      = [i for i, c in enumerate(configs) if c in open_category]

    years = sorted({y for y in instance_year.values()})

    def oracle_per_instance(idxs):
        return scores[idxs, :].max(axis=0)  # shape (n_instances,)

    def aggregate(per_inst):
        total = float(per_inst.sum())
        per_y = {y: 0.0 for y in years}
        for j, inst in enumerate(instances):
            y = instance_year.get(inst)
            if y:
                per_y[y] += float(per_inst[j])
        return total, per_y

    oracle_p_total, oracle_p_year = aggregate(oracle_per_instance(portfolio_idxs))
    oracle_a_total, oracle_a_year = aggregate(
        oracle_per_instance(portfolio_idxs + open_idxs))

    n_per_year = {y: 0 for y in years}
    for inst in instances:
        y = instance_year.get(inst)
        if y:
            n_per_year[y] += 1
    n_total = sum(n_per_year.values())
    max_per_year = {y: 15 * n for y, n in n_per_year.items()}
    max_total = 15 * n_total

    # Best single portfolio (for reference)
    best_p_total = max(float(scores[i].sum()) for i in portfolio_idxs)
    best_p_label = max(((i, float(scores[i].sum())) for i in portfolio_idxs),
                       key=lambda x: x[1])
    best_p_name = configs[best_p_label[0]][0]

    def fmt_row(label, total, per_y):
        s = f"{label:<40} {total:>9.1f}"
        for y in years:
            s += f"  {per_y[y]:>8.1f}"
        return s

    header = f"{'':<40} {'Total':>9}"
    for y in years:
        header += f"  {y:>8}"
    print(header)
    print("-" * len(header))
    print(fmt_row(f"Max possible (15 x n_instances)", max_total,
                  {y: float(max_per_year[y]) for y in years}))
    print(fmt_row(f"Oracle: portfolios + open-cat (pool 2)", oracle_a_total, oracle_a_year))
    print(fmt_row(f"Oracle: portfolios only       (pool 1)", oracle_p_total, oracle_p_year))
    print(fmt_row(f"Best single portfolio ({best_p_name})",
                  best_p_total,
                  {y: sum(float(scores[best_p_label[0], j])
                          for j, inst in enumerate(instances)
                          if instance_year.get(inst) == y)
                   for y in years}))
    print()
    print(f"Instances per year: " +
          ", ".join(f"{y}={n_per_year[y]}" for y in years) +
          f"  (total {n_total})")
    print(f"Pool 1 oracle: {oracle_p_total:.1f} / {max_total} = "
          f"{100*oracle_p_total/max_total:.1f}%")
    print(f"Pool 2 oracle: {oracle_a_total:.1f} / {max_total} = "
          f"{100*oracle_a_total/max_total:.1f}%")


if __name__ == "__main__":
    main()
