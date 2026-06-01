#!/usr/bin/env python3
"""Combined table: AI's Borda vs cpsat split by what the AI predicted.

For each AI (svc-k1, svc-ek1), shows three rows per year-block:
  * AI picks cpsat        -> measures feature-extraction overhead cost
  * AI picks alt          -> measures decision quality of the alt picks
  * combined              -> sum of the two; matches the main vs-cpsat table

Reads the per-subset CSVs produced by borda_when_ai_picks_cpsat.py and
borda_when_ai_picks_alt.py, so those must be regenerated first if the
underlying data changed (build.sh does that as part of step 5).
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CPSAT_CSV = ROOT / "leaderboard_when_cpsat.csv"
ALT_CSV   = ROOT / "leaderboard_when_alt.csv"
OUT_CSV   = ROOT / "decision_vs_overhead.csv"
OUT_TYP   = ROOT / "decision_vs_overhead.typ"

AIS = ("svc-k1", "svc-ek1")


def load(path):
    by_key = {}
    for r in csv.DictReader(open(path)):
        key = (r["ai"], r["year"])
        by_key[key] = {
            "n":             int(r["n_pick"] if "n_pick" in r else r.get("n_cpsat_pick", 0)),
            "borda_ai":      float(r["borda_ai"]),
            "borda_cpsat":   float(r["borda_cpsat"]),
        }
    return by_key


def main():
    cpsat = load(CPSAT_CSV)
    alt   = load(ALT_CSV)

    rows = []
    for ai in AIS:
        for year in ("2020", "2021", "2022", "TOTAL"):
            c = cpsat.get((ai, year), {"n": 0, "borda_ai": 0, "borda_cpsat": 0})
            a = alt.get  ((ai, year), {"n": 0, "borda_ai": 0, "borda_cpsat": 0})
            combined_n  = c["n"] + a["n"]
            combined_ai = c["borda_ai"] + a["borda_ai"]
            combined_cp = c["borda_cpsat"] + a["borda_cpsat"]
            for tag, d in [("cpsat", c), ("alt", a), ("combined",
                            {"n": combined_n, "borda_ai": combined_ai, "borda_cpsat": combined_cp})]:
                rows.append({
                    "ai":          ai,
                    "year":        year,
                    "ai_picked":   tag,
                    "n":           d["n"],
                    "borda_ai":    round(d["borda_ai"], 2),
                    "borda_cpsat": round(d["borda_cpsat"], 2),
                    "delta":       round(d["borda_ai"] - d["borda_cpsat"], 2),
                    "normalized":  round(d["borda_ai"] / d["n"], 4) if d["n"] else 0.0,
                })

    print(f"{'ai':<10}{'year':<7}{'picked':<10}{'n':>5}{'Borda AI':>10}"
          f"{'Borda cp':>10}{'Δ':>8}{'norm':>7}")
    for r in rows:
        print(f"{r['ai']:<10}{r['year']:<7}{r['ai_picked']:<10}"
              f"{r['n']:>5}{r['borda_ai']:>10.2f}{r['borda_cpsat']:>10.2f}"
              f"{r['delta']:>+8.2f}{r['normalized']:>7.4f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ai", "year", "ai_picked", "n", "borda_ai", "borda_cpsat",
            "delta", "normalized",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote -> {OUT_CSV.relative_to(ROOT)}")

    totals = [r for r in rows if r["year"] == "TOTAL"]
    typ = []
    typ.append("#figure(")
    typ.append("  table(")
    typ.append("    columns: 6,")
    typ.append("    align: (left, left, right, right, right, right),")
    typ.append("    table.header(")
    typ.append("      [AI], [AI picked], [n], "
               "[Borda AI], [Borda cpsat8], [#sym.Delta],")
    typ.append("    ),")

    label_for = {
        "cpsat":    "cpsat (overhead)",
        "alt":      "alt (decision)",
        "combined": "*combined*",
    }
    for i, r in enumerate(totals):
        if i > 0 and r["ai"] != totals[i - 1]["ai"]:
            typ.append("    table.hline(),")
        sign = "+" if r["delta"] > 0 else ""
        typ.append(
            f"    [{r['ai']}], [{label_for[r['ai_picked']]}], "
            f"[{r['n']}], [{r['borda_ai']:.2f}], "
            f"[{r['borda_cpsat']:.2f}], [{sign}{r['delta']:.2f}],"
        )
    typ.append("  ),")
    typ.append("  caption: ["
               "Decomposition of each AI selector's Borda gap against the "
               "`cpsat8` baseline over 2020-2022, split by what the AI "
               "predicted on each instance. The `cpsat (overhead)` row is "
               "where the AI's selection algorithmically matched the "
               "baseline -- any negative #sym.Delta there comes purely from "
               "the feature-extraction wall-clock cost the AI pays before "
               "it can hand cpsat the cores. The `alt (decision)` row is "
               "where the AI deviated from cpsat by selecting the "
               "k1/ek1 portfolio; its #sym.Delta isolates the decision "
               "quality of the AI's portfolio choice. The `combined` row "
               "is the union and matches the head-to-head vs-cpsat number "
               "for these years. The reading: nearly all of the Borda gap "
               "is overhead; the AI's portfolio decisions are essentially "
               "neutral (svc-k1) or mildly profitable (svc-ek1)."
               "],")
    typ.append("  ) <tab:decision-vs-overhead>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"Wrote -> {OUT_TYP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
