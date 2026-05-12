#!/usr/bin/env python3
"""
Score the final portfolios (cpsat8, k1-8c-8s-v1, ek1-8c-8s-v2) against the
15 open-category ladder solvers, using the aggregated combined_avg.csv.
Restricted to 2023-2025 (the years for which the ladder data exists).

Per-instance ceiling = 15 (no self-tie since the portfolios aren't in the
ladder; cp-sat 8c standalone is, but cpsat8 here is a separate run, so its
result vs cp-sat 8c-in-ladder is a real pairwise comparison, not a self-tie).

Output: leaderboard_vs_open.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

AVG_CSV   = ROOT / "combined_avg.csv"
OPEN_CSV  = ROOT.parent.parent / "open-category-benchmarks" / "combined.csv"
TYPES_CSV = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUT_CSV   = ROOT / "leaderboard_vs_open.csv"

YEARS = {"2023", "2024", "2025"}
MAX_TIME_MS = 1_200_000


def load_problem_types(path: Path) -> dict[tuple, str]:
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(path))}


def make_row(status, time_ms, objective, wrong=False):
    return {
        "status":    status or "",
        "time_ms":   float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective not in ("", None) else None,
        "wrong":     bool(wrong),
    }


EMPTY = make_row("", MAX_TIME_MS, "")


def score_against_ladder(entity_inst: dict, opp_inst: list[dict],
                          instances: list[tuple], kinds: dict) -> dict[tuple, float]:
    out = {}
    for key in instances:
        kind = kinds.get(key)
        if kind is None:
            out[key] = 0.0
            continue
        a = entity_inst.get(key, EMPTY)
        a_broken = a["wrong"] or a["status"] == "Error"
        s = 0.0
        for b in opp_inst:
            bb = b.get(key, EMPTY)
            b_broken = bb["wrong"] or bb["status"] == "Error"
            if a_broken and b_broken:
                sa = 0.0
            elif a_broken:
                sa = 0.0
            elif b_broken:
                sa = 1.0
            else:
                sa, _ = _compare(
                    a["status"], a["time_ms"], a["objective"],
                    bb["status"], bb["time_ms"], bb["objective"],
                    kind,
                )
            s += sa
        out[key] = s
    return out


def main():
    problem_types = load_problem_types(TYPES_CSV)
    open_rows = list(csv.DictReader(open(OPEN_CSV)))

    # Build the 15-ladder opponent set + per-instance results
    instances = []
    inst_year = {}
    inst_model = {}
    seen = set()
    for r in open_rows:
        if r["year"] not in YEARS:
            continue
        key = (r["problem"], r["name"])
        if key not in seen:
            seen.add(key)
            instances.append(key)
            inst_year[key] = r["year"]
            inst_model[key] = r["model"]
    kinds = {k: problem_types.get((k[0], inst_model[k])) for k in instances}

    open_cat = sorted({(r["solver"], int(r["cores"]))
                       for r in open_rows if r["open_category"] == "True"})
    opp_inst = []
    for cfg in open_cat:
        d = {}
        for r in open_rows:
            if (r["solver"], int(r["cores"])) == cfg and r["year"] in YEARS:
                d[(r["problem"], r["name"])] = make_row(
                    r["status"], r["time_ms"], r["objective"], r["wrong"] == "True")
        opp_inst.append(d)

    # Load each portfolio's per-instance row from combined_avg.csv (mean time)
    entities: dict[str, dict[tuple, dict]] = defaultdict(dict)
    coverage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in csv.DictReader(open(AVG_CSV)):
        if r["year"] not in YEARS:
            continue
        key = (r["problem"], r["name"])
        entities[r["schedule"]][key] = make_row(
            r["status"], r["time_ms_mean"], r["objective"])
        coverage[r["schedule"]][r["year"]] += 1

    n_inst = len(instances)
    years = sorted(YEARS)

    rendered = []
    for portfolio in sorted(entities):
        per_inst = score_against_ladder(entities[portfolio], opp_inst, instances, kinds)
        total = sum(per_inst.values())
        per_y = {y: 0.0 for y in years}
        for k, s in per_inst.items():
            per_y[inst_year[k]] += s
        max_possible = 15.0 * n_inst
        rendered.append({
            "portfolio":     portfolio,
            "borda_total":   round(total, 2),
            **{f"borda_{y}": round(per_y[y], 2) for y in years},
            "max_possible":  max_possible,
            "normalized":    round(total / max_possible, 4),
            "n_instances":   n_inst,
            "coverage":      "/".join(f"{y}:{coverage[portfolio][y]}" for y in years),
        })

    rendered.sort(key=lambda r: -r["normalized"])

    cols = (["portfolio", "borda_total"] + [f"borda_{y}" for y in years]
            + ["max_possible", "normalized", "n_instances", "coverage"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rendered)

    print(f"Years: {years}, total instances: {n_inst}")
    print()
    print(f"{'portfolio':<18} {'borda':>8} {'norm':>6}  " +
          "  ".join(f"{y:>7}" for y in years) +
          f"   max={int(15 * n_inst)}")
    for r in rendered:
        print(f"{r['portfolio']:<18} {r['borda_total']:>8.2f} {r['normalized']:>6.4f}  " +
              "  ".join(f"{r[f'borda_{y}']:>7.2f}" for y in years) +
              f"   coverage={r['coverage']}")
    print()
    print(f"Wrote -> {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
