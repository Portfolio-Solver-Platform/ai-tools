"""Direct head-to-head Borda: each AI vs its corresponding static
alternative portfolio, on 2020-2022.

Pairings:
  svc-k1   vs  k1-8c-8s-v1  (ineligible track)
  svc-ek1  vs  ek1-8c-8s-v2 (eligible track)

Evaluation set is restricted to the 283 instances also used by
borda_vs_cpsat_with_static.py (intersection of AI runs and cpsat
baseline runs), for direct comparability with that table.

Outputs:
  leaderboard_ai_vs_static.csv
  leaderboard_ai_vs_static.typ
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

AI_MEDIAN_CSV     = ROOT / "combined_median.csv"
PROD_MEDIAN_CSV   = ROOT.parent / "final-portfolios" / "combined_median.csv"
TYPES_CSV         = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUT_CSV           = ROOT / "leaderboard_ai_vs_static.csv"
OUT_TYP           = ROOT / "leaderboard_ai_vs_static.typ"

YEARS    = ("2020", "2021", "2022")
CPSAT_ID = "cpsat8"
PAIRINGS = (
    # (ai_schedule, static_schedule, track_label)
    ("svc-k1",  "k1-8c-8s-v1",  "ineligible"),
    ("svc-ek1", "ek1-8c-8s-v2", "eligible"),
)
MAX_TIME_MS = 1_200_000


def load_problem_types(path):
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(path))}


def make_row(status, time_ms, objective):
    return {
        "status":    status or "",
        "time_ms":   float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective not in ("", None) else None,
    }


def load_median(path, schedule_filter):
    out, models = {}, {}
    for r in csv.DictReader(open(path)):
        if r["schedule"] not in schedule_filter or r["year"] not in YEARS:
            continue
        key = (r["schedule"], r["year"], r["problem"], r["name"])
        out[key] = make_row(r["status"], r["time_ms"], r["objective"])
        models[(r["year"], r["problem"], r["name"])] = r["model"]
    return out, models


def main():
    types = load_problem_types(TYPES_CSV)
    ai_schedules     = {p[0] for p in PAIRINGS}
    static_schedules = {p[1] for p in PAIRINGS}

    ai_rows,   ai_model   = load_median(AI_MEDIAN_CSV,   ai_schedules)
    prod_rows, prod_model = load_median(PROD_MEDIAN_CSV, static_schedules | {CPSAT_ID})

    # Eval set per track: same 283-instance set as tab:vs-cpsat-with-static
    # (AI ran AND cpsat baseline ran).
    cpsat_keys = {(y, p, n)
                  for (sched, y, p, n) in prod_rows
                  if sched == CPSAT_ID}
    track_eval_set = {
        ai_sched: {(y, p, n) for (s, y, p, n) in ai_rows
                   if s == ai_sched and (y, p, n) in cpsat_keys}
        for ai_sched, _, _ in PAIRINGS
    }

    rows_out = []
    print(f"{'AI':<10}{'static':<16}{'year':<6}"
          f"{'borda_ai':>10}{'borda_static':>14}{'n':>5}")

    for ai_sched, static_sched, track in PAIRINGS:
        eval_set = track_eval_set[ai_sched]
        ai_running, st_running, n_running = 0.0, 0.0, 0
        for year in YEARS:
            ai_total, st_total, n_used = 0.0, 0.0, 0
            for (s, y, p, n), ai_r in ai_rows.items():
                if s != ai_sched or y != year:
                    continue
                if (y, p, n) not in eval_set:
                    continue
                static_key = (static_sched, y, p, n)
                if static_key not in prod_rows:
                    continue
                st_r = prod_rows[static_key]
                model = ai_model.get((year, p, n)) or prod_model.get((year, p, n))
                kind = types.get((p, model)) if model else None
                if kind is None:
                    continue
                sa, sb = _compare(
                    ai_r["status"], ai_r["time_ms"], ai_r["objective"],
                    st_r["status"], st_r["time_ms"], st_r["objective"],
                    kind,
                )
                ai_total += sa
                st_total += sb
                n_used   += 1
            rows_out.append({
                "ai":           ai_sched,
                "static":       static_sched,
                "track":        track,
                "year":         year,
                "borda_ai":     round(ai_total, 2),
                "borda_static": round(st_total, 2),
                "n":            n_used,
            })
            ai_running += ai_total
            st_running += st_total
            n_running  += n_used
            print(f"{ai_sched:<10}{static_sched:<16}{year:<6}"
                  f"{ai_total:>10.2f}{st_total:>14.2f}{n_used:>5}")
        rows_out.append({
            "ai":           ai_sched,
            "static":       static_sched,
            "track":        track,
            "year":         "TOTAL",
            "borda_ai":     round(ai_running, 2),
            "borda_static": round(st_running, 2),
            "n":            n_running,
        })
        print(f"{ai_sched:<10}{static_sched:<16}{'TOTAL':<6}"
              f"{ai_running:>10.2f}{st_running:>14.2f}{n_running:>5}")
        print()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ai", "static", "track", "year",
            "borda_ai", "borda_static", "n",
        ])
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote -> {OUT_CSV.relative_to(ROOT)}")

    totals = [r for r in rows_out if r["year"] == "TOTAL"]
    typ = []
    typ.append("#figure(")
    typ.append("  table(")
    typ.append("    columns: 5,")
    typ.append("    align: (left, left, right, right, right),")
    typ.append("    table.header(")
    typ.append("      [AI], [Static alternative], "
               "[Borda score AI], [Borda score static], [n],")
    typ.append("    ),")
    for r in totals:
        typ.append(
            f"    [{r['ai']}], [{r['static']}], "
            f"[{r['borda_ai']:.2f}], [{r['borda_static']:.2f}], [{r['n']}],"
        )
    typ.append("  ),")
    typ.append("  caption: ["
               "Direct head-to-head Borda score between each AI selector "
               "and its corresponding static alternative portfolio on the "
               "2020-2022 evaluation set. Each instance is scored pairwise: "
               "the better solver wins outright, otherwise the point is "
               "split by wall-clock time."
               "],")
    typ.append("  ) <tab:ai-vs-static>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"Wrote -> {OUT_TYP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
