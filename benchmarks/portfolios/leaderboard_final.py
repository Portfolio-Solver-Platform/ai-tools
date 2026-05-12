"""
Separate leaderboard for the final-portfolios re-run
(cpsat8, ek1-8c-8s-v2, k1-8c-8s-v1) scored against the 15 open-category
solvers on years 2023-2025.

Reads from final-portfolios/backup/<config>/<config>-<year>/results.csv (the
aggregated CSVs produced from the .out files in portfolios-final/).

Status translation: this dataset uses 'Unknown' with a non-empty objective
to mean 'feasible solution found, optimality not proved' — translated to
'Satisfied' so the standard Borda scoring treats it as solved.

Output: benchmarks/portfolios/leaderboard_final.csv
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
SCHED_DIRS = [
    Path("/Users/sofus/speciale/ai/ucloud-benchmark/schedules"),
    Path("/Users/sofus/speciale/ai/ucloud-benchmark/schedules-eligible"),
]
MAIN_LB_CSV = ROOT / "benchmarks/portfolios/leaderboard.csv"
OUT_CSV     = ROOT / "benchmarks/portfolios/leaderboard_final.csv"

CONFIGS = ["cpsat8", "ek1-8c-8s-v2", "k1-8c-8s-v1"]
YEARS = ["2023", "2024", "2025"]
INELIGIBLE_SOLVERS = {"solutions.huub", "org.chuffed.chuffed", "org.gecode.gecode"}
MAX_TIME_MS = 1_200_000


def load_problem_types(path):
    return {(r["problem"], r["model"]): r["type"] for r in csv.DictReader(open(path))}


def load_schedules():
    schedules = {}
    for d in SCHED_DIRS:
        for f in sorted(d.glob("*.csv")):
            if f.stem == "template":
                continue
            schedules[f.stem] = [(row[0], int(row[1]))
                                 for row in csv.reader(open(f)) if row and row[0]]
    # cpsat8 has no schedule file; encode it explicitly.
    schedules.setdefault("cpsat8", [("cp-sat", 8)])
    return schedules


def translate_status(optimal_field, objective_field):
    """Backup CSVs use 'optimal' column with values {Optimal, Unknown, Unsat}.
    Map to the standard status set; 'Unknown'+objective -> 'Satisfied'."""
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


def score_entity(entity_inst, opp_inst, instances, kinds):
    per_inst = {}
    for key in instances:
        kind = kinds.get(key)
        if kind is None:
            per_inst[key] = 0.0
            continue
        a = entity_inst.get(key, EMPTY)
        s = 0.0
        for opp in opp_inst:
            b = opp.get(key, EMPTY)
            a_broken = a["wrong"] or a["status"] == "Error"
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
        per_inst[key] = s
    return per_inst


def main():
    problem_types = load_problem_types(TYPES_CSV)
    open_rows = list(csv.DictReader(open(OPEN_CSV)))

    instances, inst_year, inst_model = [], {}, {}
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

    schedules = load_schedules()

    # Load each final-portfolios entity's per-instance results.
    entities = {}
    coverage = {}
    for cfg in CONFIGS:
        per_inst = {}
        for y in YEARS:
            csv_path = BACKUP_DIR / cfg / f"{cfg}-{y}" / "results.csv"
            if not csv_path.exists():
                continue
            for r in csv.DictReader(open(csv_path)):
                key = (r["problem"], r["name"])
                status = translate_status(r.get("optimal", ""), r.get("objective", ""))
                per_inst[key] = make_row(status, r["time_ms"], r["objective"])
        entities[cfg] = per_inst
        coverage[cfg] = {y: sum(1 for k in per_inst if inst_year.get(k) == y) for y in YEARS}

    n_inst = len(instances)
    rendered = []
    for cfg in CONFIGS:
        per_inst = score_entity(entities[cfg], opp_inst, instances, kinds)
        total = sum(per_inst.values())
        per_y = {y: 0.0 for y in YEARS}
        for k, s in per_inst.items():
            per_y[inst_year[k]] += s
        constituents = schedules.get(cfg, [])
        eligible = not any(s in INELIGIBLE_SOLVERS for s, _ in constituents)
        cstr = ",".join(f"{s}@{c}c" for s, c in constituents)
        max_possible = 15.0 * n_inst
        rendered.append({
            "name": cfg,
            "eligible": eligible,
            "borda_total": round(total, 2),
            **{f"borda_{y}": round(per_y[y], 2) for y in YEARS},
            "max_possible": max_possible,
            "normalized": round(total / max_possible, 4),
            "n_instances": n_inst,
            "coverage": "/".join(f"{y}:{coverage[cfg][y]}" for y in YEARS),
            "constituent_solvers": cstr,
        })

    rendered.sort(key=lambda r: -r["normalized"])

    cols = ["name", "eligible", "borda_total"] + \
           [f"borda_{y}" for y in YEARS] + \
           ["max_possible", "normalized", "n_instances", "coverage", "constituent_solvers"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rendered:
            w.writerow(r)

    # Prior-run comparison from the main leaderboard.
    prior = {}
    if MAIN_LB_CSV.exists():
        for r in csv.DictReader(open(MAIN_LB_CSV)):
            if r["type"] == "solver" and r["name"] == "cp-sat" and r["cores"] == "8":
                prior["cpsat8"] = (float(r["borda_total"]), float(r["normalized"]))
            elif r["type"] == "portfolio" and r["name"] in CONFIGS:
                prior[r["name"]] = (float(r["borda_total"]), float(r["normalized"]))

    print(f"Wrote {len(rendered)} rows -> {OUT_CSV.relative_to(ROOT)}")
    print(f"Years: {YEARS}, total instances: {n_inst}")
    print()
    print(f"{'name':<18} {'borda':>9}  {'norm':>6}   prior_borda  prior_norm  diff_borda  diff_norm  coverage")
    for r in rendered:
        pb, pn = prior.get(r["name"], (None, None))
        if pb is not None:
            db = r["borda_total"] - pb
            dn = r["normalized"] - pn
            print(f"{r['name']:<18} {r['borda_total']:>9.2f}  {r['normalized']:.4f}  "
                  f"{pb:>11.2f}  {pn:.4f}     {db:+8.2f}    {dn:+.4f}   {r['coverage']}")
        else:
            print(f"{r['name']:<18} {r['borda_total']:>9.2f}  {r['normalized']:.4f}  "
                  f"{'(no prior)':>11}                                    {r['coverage']}")


if __name__ == "__main__":
    main()
