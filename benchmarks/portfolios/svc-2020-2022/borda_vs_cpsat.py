#!/usr/bin/env python3
"""Head-to-head Borda: each AI vs cpsat8, per year.

Compares the median rep of each AI (from this folder's combined_median.csv)
against the median rep of cpsat8 (from final-portfolios/combined_median.csv)
on every (year, problem, name) the two share. Per-instance ceiling is 1.

Output: leaderboard_vs_cpsat.csv with one row per (ai, year) and an overall
row, plus per-year breakdown columns. Console table mirrors the CSV.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
from utils.borda import _compare, _parse_obj  # noqa: E402

AI_MEDIAN_CSV       = ROOT / "combined_median.csv"
CPSAT_MEDIAN_CSV    = ROOT.parent / "final-portfolios" / "combined_median.csv"
TYPES_CSV           = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUT_CSV             = ROOT / "leaderboard_vs_cpsat.csv"
OUT_TYP             = ROOT / "leaderboard_vs_cpsat.typ"

YEARS    = ("2020", "2021", "2022")
CPSAT_ID = "cpsat8"
AIS      = ("svc-k1", "svc-ek1")
MAX_TIME_MS = 1_200_000


def load_problem_types(path: Path) -> dict[tuple, str]:
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(path))}


def make_row(status, time_ms, objective):
    return {
        "status":    status or "",
        "time_ms":   float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective not in ("", None) else None,
    }


def load_median(path: Path, schedule_filter: set[str]) -> tuple[dict, dict]:
    by_inst: dict[tuple, dict] = {}
    inst_model: dict[tuple, str] = {}
    for r in csv.DictReader(open(path)):
        if r["schedule"] not in schedule_filter:
            continue
        if r["year"] not in YEARS:
            continue
        key = (r["schedule"], r["year"], r["problem"], r["name"])
        by_inst[key] = make_row(r["status"], r["time_ms"], r["objective"])
        inst_model[(r["year"], r["problem"], r["name"])] = r["model"]
    return by_inst, inst_model


def main():
    problem_types = load_problem_types(TYPES_CSV)

    ai_rows,    ai_model    = load_median(AI_MEDIAN_CSV, set(AIS))
    cpsat_rows, cpsat_model = load_median(CPSAT_MEDIAN_CSV, {CPSAT_ID})

    rows_out = []
    print(f"{'ai':<10} {'year':<6} {'borda_ai':>8} {'borda_cpsat':>11} "
          f"{'max':>6} {'norm':>6} {'n':>5}")

    for ai in AIS:
        ai_running    = 0.0
        cpsat_running = 0.0
        n_running     = 0
        for year in YEARS:
            shared = []
            for (sched, y, p, n), ai_r in ai_rows.items():
                if sched != ai or y != year:
                    continue
                cp_key = (CPSAT_ID, y, p, n)
                if cp_key not in cpsat_rows:
                    continue
                shared.append((p, n, ai_r, cpsat_rows[cp_key]))

            ai_total = 0.0
            cp_total = 0.0
            for p, n, ai_r, cp_r in shared:
                model = ai_model.get((year, p, n)) or cpsat_model.get((year, p, n))
                kind = problem_types.get((p, model)) if model else None
                if kind is None:
                    continue
                sa, sb = _compare(
                    ai_r["status"], ai_r["time_ms"], ai_r["objective"],
                    cp_r["status"], cp_r["time_ms"], cp_r["objective"],
                    kind,
                )
                ai_total += sa
                cp_total += sb

            n_shared = len(shared)
            max_total = float(n_shared)
            norm = ai_total / max_total if max_total else 0.0
            rows_out.append({
                "ai":           ai,
                "year":         year,
                "borda_ai":     round(ai_total, 2),
                "borda_cpsat":  round(cp_total, 2),
                "max_possible": int(max_total),
                "normalized":   round(norm, 4),
                "n_instances":  n_shared,
            })
            ai_running    += ai_total
            cpsat_running += cp_total
            n_running     += n_shared
            print(f"{ai:<10} {year:<6} {ai_total:>8.2f} {cp_total:>11.2f} "
                  f"{int(max_total):>6} {norm:>6.4f} {n_shared:>5}")

        norm = ai_running / n_running if n_running else 0.0
        rows_out.append({
            "ai":           ai,
            "year":         "TOTAL",
            "borda_ai":     round(ai_running, 2),
            "borda_cpsat":  round(cpsat_running, 2),
            "max_possible": n_running,
            "normalized":   round(norm, 4),
            "n_instances":  n_running,
        })
        print(f"{ai:<10} {'TOTAL':<6} {ai_running:>8.2f} {cpsat_running:>11.2f} "
              f"{n_running:>6} {norm:>6.4f} {n_running:>5}")
        print()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ai", "year", "borda_ai", "borda_cpsat",
            "max_possible", "normalized", "n_instances",
        ])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote -> {OUT_CSV.relative_to(ROOT)}")

    typ = []
    typ.append("#figure(")
    typ.append("  table(")
    typ.append("    columns: 6,")
    typ.append("    align: (left, right, right, right, right, right),")
    typ.append("    table.header(")
    typ.append("      [AI], [Year], [Borda AI], [Borda cpsat8], [Norm.], [n],")
    typ.append("    ),")
    for i, r in enumerate(rows_out):
        if i > 0 and r["ai"] != rows_out[i - 1]["ai"]:
            typ.append("    table.hline(),")
        year_label = r["year"] if r["year"] != "TOTAL" else "*total*"
        typ.append(
            f"    [{r['ai']}], [{year_label}], "
            f"[{r['borda_ai']:.2f}], [{r['borda_cpsat']:.2f}], "
            f"[{r['normalized']:.4f}], [{r['n_instances']}],"
        )
    typ.append("  ),")
    typ.append("  caption: ["
               "Head-to-head MZN-Challenge Borda of each AI selector "
               "(`svc-k1`, `svc-ek1`) against the `cpsat8` baseline, broken "
               "down by held-out year. Each row uses the intersection of "
               "instances both submissions ran. The per-instance maximum is "
               "1 (winner takes the full point, ties split by wall-clock "
               "time), so $text(\"Borda AI\") + text(\"Borda cpsat8\") = n$ "
               "in every row. Normalised column is $text(\"Borda AI\") / n$."
               "],")
    typ.append("  ) <tab:vs-cpsat>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"Wrote -> {OUT_TYP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
