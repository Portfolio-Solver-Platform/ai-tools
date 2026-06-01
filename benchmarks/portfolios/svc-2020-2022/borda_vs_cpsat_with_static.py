"""Head-to-head Borda against cpsat8 for 4 submissions on 2020-2022:
  svc-k1, k1-8c-8s-v1, svc-ek1, ek1-8c-8s-v2.

Lets us see whether each AI selector beats its corresponding static
alt portfolio.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

AI_MEDIAN_CSV       = ROOT / "combined_median.csv"
PROD_MEDIAN_CSV     = ROOT.parent / "final-portfolios" / "combined_median.csv"
TYPES_CSV           = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUT_CSV             = ROOT / "leaderboard_vs_cpsat_with_static.csv"
OUT_TYP             = ROOT / "leaderboard_vs_cpsat_with_static.typ"

YEARS    = ("2020", "2021", "2022")
CPSAT_ID = "cpsat8"
AI_SCHEDULES   = ("svc-k1", "svc-ek1")
PROD_SCHEDULES = ("k1-8c-8s-v1", "ek1-8c-8s-v2")
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
    ai_rows,   ai_model   = load_median(AI_MEDIAN_CSV,   set(AI_SCHEDULES))
    prod_rows, prod_model = load_median(PROD_MEDIAN_CSV, set(PROD_SCHEDULES) | {CPSAT_ID})

    cpsat_rows = {(y, p, n): row
                  for (sched, y, p, n), row in prod_rows.items()
                  if sched == CPSAT_ID}

    track_instances = {
        "svc-k1":  {(y, p, n) for (s, y, p, n) in ai_rows
                    if s == "svc-k1" and (y, p, n) in cpsat_rows},
        "svc-ek1": {(y, p, n) for (s, y, p, n) in ai_rows
                    if s == "svc-ek1" and (y, p, n) in cpsat_rows},
    }

    rows_out = []
    print(f"{'submission':<16}{'year':<6}{'borda_sub':>10}"
          f"{'borda_cpsat':>12}{'n':>5}")

    submissions = [
        ("svc-k1",       ai_rows,   "ai",     "svc-k1"),
        ("k1-8c-8s-v1",  prod_rows, "static", "svc-k1"),
        ("svc-ek1",      ai_rows,   "ai",     "svc-ek1"),
        ("ek1-8c-8s-v2", prod_rows, "static", "svc-ek1"),
    ]
    for sched, rows, kind, track in submissions:
        eval_set = track_instances[track]
        sub_running, cp_running, n_running = 0.0, 0.0, 0
        for year in YEARS:
            sub_total, cp_total, n_used = 0.0, 0.0, 0
            for (s, y, p, n), sub_r in rows.items():
                if s != sched or y != year:
                    continue
                if (y, p, n) not in eval_set:
                    continue
                cp = cpsat_rows[(y, p, n)]
                model = ai_model.get((year, p, n)) or prod_model.get((year, p, n))
                kind_p = types.get((p, model)) if model else None
                if kind_p is None:
                    continue
                sa, sb = _compare(
                    sub_r["status"], sub_r["time_ms"], sub_r["objective"],
                    cp["status"],    cp["time_ms"],    cp["objective"],
                    kind_p,
                )
                sub_total += sa
                cp_total  += sb
                n_used    += 1
            rows_out.append({
                "submission":  sched,
                "kind":        kind,
                "year":        year,
                "borda_sub":   round(sub_total, 2),
                "borda_cpsat": round(cp_total, 2),
                "n":           n_used,
            })
            sub_running += sub_total
            cp_running  += cp_total
            n_running   += n_used
            print(f"{sched:<16}{year:<6}{sub_total:>10.2f}"
                  f"{cp_total:>12.2f}{n_used:>5}")
        rows_out.append({
            "submission":  sched,
            "kind":        kind,
            "year":        "TOTAL",
            "borda_sub":   round(sub_running, 2),
            "borda_cpsat": round(cp_running, 2),
            "n":           n_running,
        })
        print(f"{sched:<16}{'TOTAL':<6}{sub_running:>10.2f}"
              f"{cp_running:>12.2f}{n_running:>5}")
        print()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "submission", "kind", "year", "borda_sub", "borda_cpsat", "n",
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
    typ.append("      [Submission], [Type], [Borda], [Borda cpsat8], [n],")
    typ.append("    ),")
    type_label = {"ai": "AI", "static": "static alt"}
    for i, r in enumerate(totals):
        if i > 0 and i % 2 == 0:
            typ.append("    table.hline(),")
        typ.append(
            f"    [{r['submission']}], [{type_label[r['kind']]}], "
            f"[{r['borda_sub']:.2f}], [{r['borda_cpsat']:.2f}], [{r['n']}],"
        )
    typ.append("  ),")
    typ.append("  caption: ["
               "Head-to-head Borda against the #cpsat() baseline on the "
               "2020-2022 evaluation set, for each AI selector and its "
               "corresponding static alternative portfolio. Comparing the "
               "AI row to the static-alt row in each track shows whether "
               "the AI's selection beats running the alt portfolio "
               "unconditionally."
               "],")
    typ.append("  ) <tab:vs-cpsat-with-static>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"Wrote -> {OUT_TYP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
