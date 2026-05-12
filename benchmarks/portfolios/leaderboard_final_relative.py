"""
Relative leaderboard: 4 entities scored only against each other.
  - cpsat8         (final-portfolios re-run)
  - ek1-8c-8s-v2   (final-portfolios re-run)
  - k1-8c-8s-v1    (final-portfolios re-run)
  - cp-sat(8c)     (original open-category data)

Restricted to the 295 instances all four entities cover (the 5 instances
missing from the 2025 final-portfolios run are skipped).

Per-instance ceiling = 3, total ceiling = 3 * 295 = 885.

Output: benchmarks/portfolios/leaderboard_final_relative.csv
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.borda import _compare, _parse_obj  # noqa: E402

OPEN_CSV   = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV  = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
BACKUP_DIR = ROOT / "benchmarks/portfolios/final-portfolios/backup"
OUT_CSV    = ROOT / "benchmarks/portfolios/leaderboard_final_relative.csv"

FINAL_CONFIGS = ["cpsat8", "ek1-8c-8s-v2", "k1-8c-8s-v1"]
OPEN_CPSAT = ("cp-sat", 8)
YEARS = ["2023", "2024", "2025"]
MAX_TIME_MS = 1_200_000


def load_problem_types(path):
    return {(r["problem"], r["model"]): r["type"] for r in csv.DictReader(open(path))}


def translate_status(optimal_field, objective_field):
    if optimal_field == "Optimal":
        return "Optimal"
    if optimal_field == "Unsat":
        return "Unsat"
    if optimal_field == "Unknown":
        return "Satisfied" if (objective_field not in ("", None)) else "Unknown"
    return optimal_field or ""


def make_row(status, time_ms, objective, wrong=False):
    return {
        "status": status or "",
        "time_ms": float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective else None,
        "wrong": bool(wrong),
    }


EMPTY = make_row("", MAX_TIME_MS, "")


def main():
    problem_types = load_problem_types(TYPES_CSV)
    open_rows = list(csv.DictReader(open(OPEN_CSV)))

    # Open-cat universe: 300 instances + their year/model.
    instances_300 = []
    inst_year = {}
    inst_model = {}
    seen = set()
    for r in open_rows:
        if r["year"] not in YEARS:
            continue
        key = (r["problem"], r["name"])
        if key not in seen:
            seen.add(key)
            instances_300.append(key)
            inst_year[key] = r["year"]
            inst_model[key] = r["model"]
    kinds = {k: problem_types.get((k[0], inst_model[k])) for k in instances_300}

    # cp-sat(8c) per-instance rows from open-cat data.
    cpsat_open = {}
    for r in open_rows:
        if (r["solver"], int(r["cores"])) == OPEN_CPSAT and r["year"] in YEARS:
            cpsat_open[(r["problem"], r["name"])] = make_row(
                r["status"], r["time_ms"], r["objective"], r["wrong"] == "True")

    # Three final-portfolios entities from backup CSVs.
    final = {cfg: {} for cfg in FINAL_CONFIGS}
    for cfg in FINAL_CONFIGS:
        for y in YEARS:
            csv_path = BACKUP_DIR / cfg / f"{cfg}-{y}" / "results.csv"
            if not csv_path.exists():
                continue
            for r in csv.DictReader(open(csv_path)):
                key = (r["problem"], r["name"])
                status = translate_status(r.get("optimal", ""), r.get("objective", ""))
                final[cfg][key] = make_row(status, r["time_ms"], r["objective"])

    # Restrict to instances all 4 entities cover.
    common = [k for k in instances_300
              if k in cpsat_open and all(k in final[c] for c in FINAL_CONFIGS)]
    print(f"Common instances across all 4 entities: {len(common)} / {len(instances_300)}")

    # Build entity list: name -> (label, dict)
    entities = [
        ("cp-sat(8c)-open", cpsat_open),
        *[(c, final[c]) for c in FINAL_CONFIGS],
    ]

    # Score each entity against the other 3.
    n_inst = len(common)
    n_opp = len(entities) - 1
    per_inst_total = {label: {} for label, _ in entities}
    for key in common:
        kind = kinds.get(key)
        if kind is None:
            for label, _ in entities:
                per_inst_total[label][key] = 0.0
            continue
        rows = [d.get(key, EMPTY) for _, d in entities]
        for i, (label, _) in enumerate(entities):
            a = rows[i]
            a_broken = a["wrong"] or a["status"] == "Error"
            s = 0.0
            for j, b in enumerate(rows):
                if i == j:
                    continue
                b_broken = b["wrong"] or b["status"] == "Error"
                if a_broken and b_broken:
                    sa = 0.0
                elif a_broken:
                    sa = 0.0
                elif b_broken:
                    sa = 1.0
                else:
                    sa, _ = _compare(
                        a["status"], a["time_ms"], a["objective"],
                        b["status"], b["time_ms"], b["objective"],
                        kind,
                    )
                s += sa
            per_inst_total[label][key] = s

    rendered = []
    for label, _ in entities:
        per_inst = per_inst_total[label]
        total = sum(per_inst.values())
        per_y = {y: 0.0 for y in YEARS}
        n_per_y = {y: 0 for y in YEARS}
        for k, s in per_inst.items():
            per_y[inst_year[k]] += s
            n_per_y[inst_year[k]] += 1
        max_possible = n_opp * n_inst
        rendered.append({
            "name": label,
            "borda_total": round(total, 2),
            **{f"borda_{y}": round(per_y[y], 2) for y in YEARS},
            "max_possible": max_possible,
            "normalized": round(total / max_possible, 4),
            "n_instances": n_inst,
            "instances_per_year": "/".join(f"{y}:{n_per_y[y]}" for y in YEARS),
        })

    rendered.sort(key=lambda r: -r["normalized"])

    cols = ["name", "borda_total"] + \
           [f"borda_{y}" for y in YEARS] + \
           ["max_possible", "normalized", "n_instances", "instances_per_year"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rendered:
            w.writerow(r)

    print(f"Wrote {len(rendered)} rows -> {OUT_CSV.relative_to(ROOT)}")
    print(f"Per-instance ceiling = {n_opp}, total = {n_opp * n_inst}")
    print()
    print(f"{'name':<22} {'borda':>8} {'norm':>7}   2023      2024      2025")
    for r in rendered:
        print(f"{r['name']:<22} {r['borda_total']:>8.2f} {r['normalized']:>7.4f}   "
              f"{r['borda_2023']:>6.2f}    {r['borda_2024']:>6.2f}    {r['borda_2025']:>6.2f}")


if __name__ == "__main__":
    main()
