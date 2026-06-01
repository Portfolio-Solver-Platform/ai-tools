from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN_CSV = ROOT / "out" / "year_leakage.csv"
OUT_TYP = ROOT / "out" / "year_leakage.typ"


def main() -> None:
    rows = []
    with open(IN_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    years   = [int(r["year"]) for r in rows]
    strict  = [float(r["strict_pct"]) for r in rows]
    model_  = [float(r["model_pct"])  for r in rows]
    family  = [float(r["family_pct"]) for r in rows]

    n = len(years)
    tick_pairs = "(" + ", ".join(f"({i}, [{y}])" for i, y in enumerate(years)) + ")"
    y_max = max(max(family), 100.0) * 1.05

    def fl(vs):
        return "(" + ", ".join(f"{v:.2f}" for v in vs) + ",)"

    typ = []
    typ.append("#figure(")
    typ.append("  {")
    typ.append(f"    let n = {n}")
    typ.append("    let xs = range(n).map(i => float(i))")
    typ.append("    lq.diagram(")
    typ.append("      width: 16cm, height: 7cm,")
    typ.append("      title: [Year-by-year instance leakage under leave-one-year-out],")
    typ.append("      xlabel: [Held-out year],")
    typ.append("      ylabel: [Share of held-out instances also in training (%)],")
    typ.append(f"      xaxis: (ticks: {tick_pairs}, subticks: none),")
    typ.append(f"      ylim: (0, {y_max:.1f}),")
    typ.append(f"      lq.bar(xs.map(x => x - 0.28), {fl(strict)}, "
               f"width: 0.25, fill: red.lighten(20%), label: [same instance (.mzn + .dzn)]),")
    typ.append(f"      lq.bar(xs.map(x => x), {fl(model_)}, "
               f"width: 0.25, fill: orange.lighten(20%), label: [same .mzn]),")
    typ.append(f"      lq.bar(xs.map(x => x + 0.28), {fl(family)}, "
               f"width: 0.25, fill: blue.lighten(20%), label: [same problem family]),")
    typ.append("    )")
    typ.append("  },")
    typ.append("  caption: ["
               "Leave-one-year-out leakage of the cpsat8/k1 training data: for each year "
               "$y$ (x-axis), the bars show the fraction of $y$'s instances whose identity "
               "also appears somewhere in the remaining 14 training years. Three identity "
               "levels are shown: same exact instance (.mzn + .dzn together), same MiniZinc "
               "model file (.mzn) but possibly different data, and same problem family "
               "(folder). The strictest level is the only one that is true data leakage; "
               "the looser levels reflect the MiniZinc Challenge's recurring problem "
               "structure that the AI is expected to generalise across. The years "
               "2023, 2024, and 2025 are off-limits as evaluation years because the "
               "k1 and ek1 portfolios were selected based on those years; the "
               "comparable leakage profile of 2020-2022 makes them legitimate stand-ins."
               "],")
    typ.append("  ) <fig:year-leakage>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"wrote {OUT_TYP}")


if __name__ == "__main__":
    main()
