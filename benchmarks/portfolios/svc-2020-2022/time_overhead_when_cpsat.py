#!/usr/bin/env python3
"""Time-overhead summary on cpsat-picked instances.

Reads cpsat_predicted_instances.csv (produced by borda_when_ai_picks_cpsat.py)
and reports, for each (AI, year) and the TOTAL, the distribution of the
wall-clock difference (AI - cpsat baseline) in seconds. These are the
instances where the AI's portfolio choice algorithmically matched the
baseline, so the delta is pure infrastructure overhead.

Output: time_overhead_when_cpsat.csv  (full per-year breakdown)
        time_overhead_when_cpsat.typ  (TOTAL-only thesis table)
"""
from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_CSV  = ROOT / "cpsat_predicted_instances.csv"
OUT_CSV = ROOT / "time_overhead_when_cpsat.csv"
OUT_TYP = ROOT / "time_overhead_when_cpsat.typ"

AIS   = ("svc-k1", "svc-ek1")
YEARS = ("2020", "2021", "2022")


def pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = max(0, min(len(s) - 1, int(q * len(s))))
    return s[i]


def summary(rows):
    deltas = [int(r["delta_ms"]) / 1000.0 for r in rows]
    slow   = [d for d in deltas if d > 0]
    n_slow = len(slow)
    n_fast = sum(1 for d in deltas if d < 0)
    n_eq   = sum(1 for d in deltas if d == 0)
    return {
        "n":              len(deltas),
        "median_s":       round(st.median(deltas), 2) if deltas else 0.0,
        "mean_s":         round(st.mean(deltas), 2)   if deltas else 0.0,
        "median_slow_s":  round(st.median(slow), 2)   if slow else 0.0,
        "mean_slow_s":    round(st.mean(slow), 2)     if slow else 0.0,
        "p90_s":          round(pct(deltas, 0.90), 2),
        "p95_s":          round(pct(deltas, 0.95), 2),
        "max_s":          round(max(deltas), 2) if deltas else 0.0,
        "frac_ai_slow":   round(n_slow / len(deltas), 3) if deltas else 0.0,
        "frac_ai_fast":   round(n_fast / len(deltas), 3) if deltas else 0.0,
        "frac_equal":     round(n_eq   / len(deltas), 3) if deltas else 0.0,
    }


def main():
    all_rows = list(csv.DictReader(open(IN_CSV)))
    groups = defaultdict(list)
    for r in all_rows:
        groups[(r["ai"], r["year"])].append(r)
        groups[(r["ai"], "TOTAL")].append(r)

    rows_out = []
    print(f"{'ai':<10}{'year':<7}{'n':>4}{'med Δ':>8}{'med Δ|slow':>11}"
          f"{'mean Δ|slow':>12}{'p90':>7}{'p95':>7}{'max':>9}"
          f"{'%slow':>7}{'%fast':>7}{'%==':>6}")
    for ai in AIS:
        for year in (*YEARS, "TOTAL"):
            s = summary(groups[(ai, year)])
            rows_out.append({"ai": ai, "year": year, **s})
            print(f"{ai:<10}{year:<7}{s['n']:>4}{s['median_s']:>8.2f}"
                  f"{s['median_slow_s']:>11.2f}{s['mean_slow_s']:>12.2f}"
                  f"{s['p90_s']:>7.2f}{s['p95_s']:>7.2f}{s['max_s']:>9.2f}"
                  f"{s['frac_ai_slow']*100:>6.1f}%"
                  f"{s['frac_ai_fast']*100:>6.1f}%"
                  f"{s['frac_equal']*100:>5.1f}%")
        print()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ai", "year", "n",
            "median_s", "mean_s", "median_slow_s", "mean_slow_s",
            "p90_s", "p95_s", "max_s",
            "frac_ai_slow", "frac_ai_fast", "frac_equal",
        ])
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote -> {OUT_CSV.relative_to(ROOT)}")

    totals = [r for r in rows_out if r["year"] == "TOTAL"]
    typ = []
    typ.append("#figure(")
    typ.append("  table(")
    typ.append("    columns: 7,")
    typ.append("    align: (left, right, right, right, right, right, right),")
    typ.append("    table.header(")
    typ.append("      [AI], [n], [med (s)], [med slowdown (s)], "
               "[mean slowdown (s)], [p90 (s)], [AI slower],")
    typ.append("    ),")
    for r in totals:
        typ.append(
            f"    [{r['ai']}], [{r['n']}], "
            f"[{r['median_s']:.2f}], "
            f"[{r['median_slow_s']:.2f}], [{r['mean_slow_s']:.2f}], "
            f"[{r['p90_s']:.2f}], "
            f"[{r['frac_ai_slow']*100:.1f}%],"
        )
    typ.append("  ),")
    typ.append("  caption: ["
               "Wall-clock difference $text(\"AI\") - text(\"cpsat8\")$ on "
               "instances where the AI predicted cpsat. *med (s)* is the "
               "median over all such instances; the two slowdown columns "
               "are the median and mean *restricted to instances where the "
               "AI was actually slower than the cpsat baseline*, since the "
               "overall mean is dragged below zero by a handful of long "
               "instances where the static portfolio solved the problem "
               "before cpsat would have. *AI slower* is the share of "
               "instances where the difference is strictly positive."
               "],")
    typ.append("  ) <tab:time-overhead-when-cpsat>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"Wrote -> {OUT_TYP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
