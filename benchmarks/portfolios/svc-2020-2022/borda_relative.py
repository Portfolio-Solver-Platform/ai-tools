#!/usr/bin/env python3
"""
Three-way Borda comparison among the final portfolios (cpsat8, k1-8c-8s-v1,
ek1-8c-8s-v2) on every instance they share.

Each portfolio competes against the other two on every instance using the
median rep's row from combined_median.csv. Per-instance ceiling is 2
(beats both others); total ceiling is 2 * n_instances.

Output: leaderboard_relative.csv, with both an "all years" view and a per-year
breakdown.
"""
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

MEDIAN_CSV = ROOT / "combined_median.csv"
TYPES_CSV = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUT_CSV = ROOT / "leaderboard_relative.csv"

MAX_TIME_MS = 1_200_000


def load_problem_types(path: Path) -> dict[tuple, str]:
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(path))}


def make_row(status, time_ms, objective):
    return {
        "status":    status or "",
        "time_ms":   float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective not in ("", None) else None,
        "wrong":     False,
    }


def pairwise(a: dict, b: dict, kind: str) -> tuple[float, float]:
    if kind is None:
        return 0.0, 0.0
    sa, sb = _compare(
        a["status"], a["time_ms"], a["objective"],
        b["status"], b["time_ms"], b["objective"],
        kind,
    )
    return sa, sb


def main():
    problem_types = load_problem_types(TYPES_CSV)

    # {(year, problem, name): {portfolio: row}}
    instances: dict[tuple, dict[str, dict]] = defaultdict(dict)
    inst_model: dict[tuple, str] = {}

    for r in csv.DictReader(open(MEDIAN_CSV)):
        key = (r["year"], r["problem"], r["name"])
        instances[key][r["schedule"]] = make_row(
            r["status"], r["time_ms"], r["objective"])
        inst_model[key] = r["model"]

    portfolios = sorted({pf for d in instances.values() for pf in d})
    print(f"Portfolios: {portfolios}")
    print(f"Instances:  {len(instances)}")

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    n_per_year: dict[str, int] = defaultdict(int)

    for (year, problem, name), per_pf in instances.items():
        if set(per_pf.keys()) != set(portfolios):
            print(f"WARN: missing portfolio data for {year}/{problem}/{name}: "
                  f"{sorted(per_pf.keys())}", file=sys.stderr)
            continue
        kind = problem_types.get((problem, inst_model[(year, problem, name)]))
        n_per_year[year] += 1
        for a_name, b_name in combinations(portfolios, 2):
            a, b = per_pf[a_name], per_pf[b_name]
            sa, sb = pairwise(a, b, kind)
            totals[a_name][year] += sa
            totals[b_name][year] += sb

    years = sorted(n_per_year)
    n_total = sum(n_per_year.values())
    n_opp = len(portfolios) - 1

    rendered = []
    for pf in portfolios:
        per_y = totals[pf]
        total = sum(per_y.values())
        max_total = n_opp * n_total
        rendered.append({
            "portfolio":     pf,
            "borda_total":   round(total, 2),
            **{f"borda_{y}": round(per_y[y], 2) for y in years},
            "max_possible":  max_total,
            "normalized":    round(total / max_total, 4) if max_total else 0,
            "n_instances":   n_total,
        })

    rendered.sort(key=lambda r: -r["normalized"])

    cols = (["portfolio", "borda_total"] + [f"borda_{y}" for y in years]
            + ["max_possible", "normalized", "n_instances"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rendered)

    print()
    print(f"{'portfolio':<18} {'borda':>8} {'norm':>6}  " +
          "  ".join(f"{y:>7}" for y in years) +
          f"   max={n_opp * n_total}")
    for r in rendered:
        print(f"{r['portfolio']:<18} {r['borda_total']:>8.2f} {r['normalized']:>6.4f}  " +
              "  ".join(f"{r[f'borda_{y}']:>7.2f}" for y in years))
    print()
    print(f"n_per_year: {dict(n_per_year)}")
    print(f"Wrote -> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
