"""
Combined leaderboard: every solver config + every portfolio (eligible and
ineligible), each scored by Borda count against the 15 open-category solvers.

Self-comparison is included (a candidate that is itself one of the 15
opponents competes against itself, which always ties).

Output: benchmarks/portfolios/leaderboard.csv
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.borda import _compare, _parse_obj  # noqa: E402

OPEN_CSV   = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV  = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
ALL_CSV    = ROOT / "benchmarks/portfolios/all/combined.csv"
ELIG_CSV   = ROOT / "benchmarks/portfolios/eligible/combined.csv"
SCHED_DIRS = [
    Path("/Users/sofus/speciale/ai/ucloud-benchmark/schedules"),
    Path("/Users/sofus/speciale/ai/ucloud-benchmark/schedules-eligible"),
]
OUT_CSV = ROOT / "benchmarks/portfolios/leaderboard.csv"

INELIGIBLE_SOLVERS = {"solutions.huub", "org.chuffed.chuffed", "org.gecode.gecode"}
MAX_TIME_MS = 1_200_000


def load_problem_types(path):
    types = {}
    for row in csv.DictReader(open(path)):
        types[(row["problem"], row["model"])] = row["type"]
    return types


def load_schedules():
    """Return {schedule_name: [(solver, cores), ...]} from both dirs (deduped)."""
    schedules = {}
    for d in SCHED_DIRS:
        for f in sorted(d.glob("*.csv")):
            name = f.stem
            if name == "template":
                continue
            constituents = []
            for row in csv.reader(open(f)):
                if row and row[0]:
                    constituents.append((row[0], int(row[1])))
            schedules[name] = constituents
    return schedules


def make_row(status, time_ms, objective, wrong):
    """Build a row dict matching what _compare expects."""
    return {
        "status": status or "",
        "time_ms": float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective else None,
        "wrong": (str(wrong) == "True"),
    }


EMPTY = make_row("", MAX_TIME_MS, "", False)


def score_entity(entity_inst, opponent_inst, instances, kinds):
    """
    entity_inst: {(problem, name): row}
    opponent_inst: list of 15 dicts, each {(problem, name): row}
    instances: list of (problem, name)
    kinds: {(problem, name): "SAT"|"MIN"|"MAX"|None}
    Returns (total, per_year_dict_by_(problem,name)->score)
    """
    per_inst = {}
    for key in instances:
        kind = kinds.get(key)
        if kind is None:
            per_inst[key] = 0.0
            continue
        a = entity_inst.get(key, EMPTY)
        s = 0.0
        for opp in opponent_inst:
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

    # Discover the 300 instances and their year/model from the open-cat csv.
    instances = []
    inst_year = {}
    inst_model = {}
    seen = set()
    for r in open_rows:
        key = (r["problem"], r["name"])
        if key not in seen:
            seen.add(key)
            instances.append(key)
            inst_year[key] = r["year"]
            inst_model[key] = r["model"]
    kinds = {k: problem_types.get((p, inst_model[k]))
             for k, (p, _) in zip(instances, instances)}

    # 15 open-cat opponent configs and their per-instance rows.
    open_cat = sorted({(r["solver"], int(r["cores"]))
                       for r in open_rows if r["open_category"] == "True"})
    opp_inst = []
    for cfg in open_cat:
        d = {}
        for r in open_rows:
            if (r["solver"], int(r["cores"])) == cfg:
                d[(r["problem"], r["name"])] = make_row(
                    r["status"], r["time_ms"], r["objective"], r["wrong"])
        opp_inst.append(d)

    # All 40 solver configs from the open-cat csv (including non-open-cat ones).
    solver_configs = sorted({(r["solver"], int(r["cores"])) for r in open_rows})
    solver_inst = {cfg: {} for cfg in solver_configs}
    for r in open_rows:
        cfg = (r["solver"], int(r["cores"]))
        solver_inst[cfg][(r["problem"], r["name"])] = make_row(
            r["status"], r["time_ms"], r["objective"], r["wrong"])

    # Portfolios: schedule definitions + per-instance results.
    schedules = load_schedules()
    portfolio_inst = {name: {} for name in schedules}
    for csv_path in (ALL_CSV, ELIG_CSV):
        for r in csv.DictReader(open(csv_path)):
            name = r["schedule"]
            if name not in portfolio_inst:
                portfolio_inst[name] = {}  # in data but not in schedule dirs
            portfolio_inst[name][(r["problem"], r["name"])] = make_row(
                r["status"], r["time_ms"], r["objective"], r["wrong"])

    # Score everything against the 15 open-cat opponents.
    rows_out = []
    for cfg in solver_configs:
        per_inst = score_entity(solver_inst[cfg], opp_inst, instances, kinds)
        rows_out.append(("solver", cfg[0], cfg[1], cfg in open_cat, "", per_inst,
                         solver_inst[cfg]))

    for name, constituents in schedules.items():
        eligible = not any(s in INELIGIBLE_SOLVERS for s, _ in constituents)
        per_inst = score_entity(portfolio_inst.get(name, {}), opp_inst, instances, kinds)
        cstr = ",".join(f"{s}@{c}c" for s, c in constituents)
        rows_out.append(("portfolio", name, "", eligible, cstr, per_inst,
                         portfolio_inst.get(name, {})))

    # All-vs-all: each entity scored against the other 82 + self-tie.
    # Per-instance ceiling = 82 + 0.5 = 82.5 → total ceiling = 82.5 * n_inst.
    all_inst_dicts = [r[6] for r in rows_out]
    n_entities = len(all_inst_dicts)
    all_v_all_totals = [0.0] * n_entities
    for key in instances:
        kind = kinds.get(key)
        if kind is None:
            continue
        rows_at_inst = [d.get(key, EMPTY) for d in all_inst_dicts]
        for i, a in enumerate(rows_at_inst):
            s = 0.0
            a_broken = a["wrong"] or a["status"] == "Error"
            for j, b in enumerate(rows_at_inst):
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
            all_v_all_totals[i] += s

    # Aggregate totals + per-year.
    # max_possible per candidate: 15 per instance for non-opponents; 14.5 for
    # the 15 open-cat configs (their self-comparison ties at 0.5, never 1.0).
    years = sorted({y for y in inst_year.values()})
    n_inst = len(instances)
    max_ava = (n_entities - 1 + 0.5) * n_inst
    rendered = []
    for i, (typ, name, cores, eligible, cstr, per_inst, _) in enumerate(rows_out):
        total = sum(per_inst.values())
        per_y = {y: 0.0 for y in years}
        for key, s in per_inst.items():
            per_y[inst_year[key]] += s
        is_self_opponent = (typ == "solver" and (name, cores) in open_cat)
        max_possible = (14.5 if is_self_opponent else 15.0) * n_inst
        ava = all_v_all_totals[i]
        rendered.append({
            "type": typ, "name": name, "cores": cores,
            "eligible": eligible,
            "borda_total": round(total, 2),
            **{f"borda_{y}": round(per_y[y], 2) for y in years},
            "max_possible": max_possible,
            "normalized": round(total / max_possible, 4),
            "borda_all_vs_all": round(ava, 2),
            "max_all_vs_all": max_ava,
            "normalized_all_vs_all": round(ava / max_ava, 4),
            "n_instances": n_inst,
            "constituent_solvers": cstr,
        })

    rendered.sort(key=lambda r: -r["borda_total"])

    cols = ["type", "name", "cores", "eligible", "borda_total"] + \
           [f"borda_{y}" for y in years] + \
           ["max_possible", "normalized",
            "borda_all_vs_all", "max_all_vs_all", "normalized_all_vs_all",
            "n_instances", "constituent_solvers"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rendered:
            w.writerow(r)

    print(f"Wrote {len(rendered)} rows -> {OUT_CSV.relative_to(ROOT)}")
    print(f"  solvers:    {sum(1 for r in rendered if r['type']=='solver')}")
    print(f"  portfolios: {sum(1 for r in rendered if r['type']=='portfolio')} "
          f"(eligible: {sum(1 for r in rendered if r['type']=='portfolio' and r['eligible'])}, "
          f"ineligible: {sum(1 for r in rendered if r['type']=='portfolio' and not r['eligible'])})")
    print(f"  instances:  {len(instances)}  (max possible per row = 15 * {len(instances)} = {15*len(instances)})")
    print()
    print("Top 10 (sorted by normalized vs 15 open-cat):")
    print(f"  {'':<2} {'name':<40} {'borda':>8} {'norm':>6}  {'ava':>9} {'norm_ava':>8}")
    for r in rendered[:10]:
        label = f"{r['name']}({r['cores']}c)" if r["type"] == "solver" else r["name"]
        tag = "S " if r["type"] == "solver" else ("Pe" if r["eligible"] else "Pi")
        print(f"  {tag} {label:<40} {r['borda_total']:>8.2f} {r['normalized']:>6.4f}  "
              f"{r['borda_all_vs_all']:>9.2f} {r['normalized_all_vs_all']:>8.4f}")


if __name__ == "__main__":
    main()
