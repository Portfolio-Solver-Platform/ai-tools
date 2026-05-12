#!/usr/bin/env python3
"""Generate a Typst table comparing 'old' (older campaign) vs 'new' (final-portfolios)
solve times for the three portfolios cpsat8, k1, ek1 over years 2023-2025.

Old sources:
  - cpsat8:  open-category combined.csv, solver=cp-sat cores=8 (the standalone
             reference, NOT parasol-wrapped — caveat noted in caption)
  - k1:      benchmarks/portfolios/all/portfolios/k1-8c-8s-v1-{year}/
  - ek1:     benchmarks/portfolios/eligible/portfolios/ek1-8c-8s-v2/ek1-8c-8s-v2-{year}/
New source:
  - all:     benchmarks/portfolios/final-portfolios/portfolios-final/{portfolio}/{portfolio}-{year}/

Comparison restricted to instances both runs solved without timing out.
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
ALL_PORTFOLIOS = ROOT / "benchmarks/portfolios/all/portfolios"
ELIGIBLE_PORTFOLIOS = ROOT / "benchmarks/portfolios/eligible/portfolios"
NEW_ROOT = ROOT / "benchmarks/portfolios/final-portfolios/portfolios-final"
OUT_FILE = Path(__file__).resolve().parent / "portfolio_comparison_table.typ"

YEARS = ("2023", "2024", "2025")


def load_cpsat_old(year):
    out = {}
    with open(OPEN_CSV) as f:
        for r in csv.DictReader(f):
            if r["solver"] == "cp-sat" and r["cores"] == "8" and r["year"] == year:
                out[(r["problem"], r["model"], r["name"])] = r
    return out


def load_csv(path):
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["problem"], r["model"], r["name"])] = r
    return out


def stats_for(old, new, old_status_key="optimal"):
    """Restrict to instances both solved cleanly. Return n, medians, means, ratio."""
    old_t, new_t, ratios = [], [], []
    for k in sorted(set(old) & set(new)):
        o, n = old[k], new[k]
        ot = float(o["time_ms"])
        nt = float(n["time_ms"])
        if ot >= 1_200_000 or nt >= 1_200_000:
            continue
        o_ok = o.get(old_status_key, "") in ("Optimal", "Unsat", "Satisfied") or o.get("objective", "")
        n_ok = n.get("optimal", "") in ("Optimal", "Unsat") or n.get("objective", "")
        if not (o_ok and n_ok):
            continue
        old_t.append(ot)
        new_t.append(nt)
        if ot > 100:
            ratios.append(nt / ot)
    if not old_t:
        return None
    return {
        "n": len(old_t),
        "med_old": statistics.median(old_t),
        "med_new": statistics.median(new_t),
        "mean_old": statistics.mean(old_t),
        "mean_new": statistics.mean(new_t),
        "med_ratio": statistics.median(ratios) if ratios else float("nan"),
    }


def fmt_ms(ms):
    return f"{ms / 1000:.2f}"


def main():
    rows = []
    for portfolio in ("cpsat8", "k1-8c-8s-v1", "ek1-8c-8s-v2"):
        for year in YEARS:
            if portfolio == "cpsat8":
                old = load_cpsat_old(year)
                old_status_key = "status"
            elif portfolio == "k1-8c-8s-v1":
                old = load_csv(ALL_PORTFOLIOS / f"{portfolio}-{year}" / "results.csv")
                old_status_key = "status"
            elif portfolio == "ek1-8c-8s-v2":
                old = load_csv(ELIGIBLE_PORTFOLIOS / portfolio / f"{portfolio}-{year}" / "results.csv")
                old_status_key = "status"

            new = load_csv(NEW_ROOT / portfolio.replace("-8c-8s-v1", "-8c-8s-v1").replace("-8c-8s-v2", "-8c-8s-v2") / f"{portfolio}-{year}" / "results.csv")
            s = stats_for(old, new, old_status_key)
            if s is None:
                continue
            rows.append({
                "portfolio": portfolio,
                "year": year,
                **s,
            })

    # Build typst table
    label = {"cpsat8": "cpsat8", "k1-8c-8s-v1": "k1", "ek1-8c-8s-v2": "ek1"}

    cells = []
    # header
    cells.extend([
        "[*Portfolio*]", "[*Year*]", "[*n*]",
        "[*Old median*]", "[*New median*]",
        "[*Old mean*]", "[*New mean*]",
        "[*Median ratio*]",
    ])
    for r in rows:
        cells.extend([
            f"[{label[r['portfolio']]}]",
            f"[{r['year']}]",
            f"[{r['n']}]",
            f"[{fmt_ms(r['med_old'])} s]",
            f"[{fmt_ms(r['med_new'])} s]",
            f"[{fmt_ms(r['mean_old'])} s]",
            f"[{fmt_ms(r['mean_new'])} s]",
            f"[{r['med_ratio']:.2f}×]",
        ])

    table_body = ",\n  ".join(cells)

    typst = f"""\
#set page(width: auto, height: auto, margin: 1em)

#table(
  columns: 8,
  align: (left, center, right, right, right, right, right, right),
  stroke: 0.5pt,

  {table_body}
)

#set par(justify: true)
*Old vs new run times for the same portfolio.* For each portfolio and year,
restrict to challenge instances that finished within the 1200 s timeout in
both runs. Old data sources:

- *cpsat8*: open-category benchmarks (`cp-sat`, 8 cores) — note this is
  cp-sat run *standalone*, not via parasol, so this row mixes harness and
  machine differences.
- *k1*: `benchmarks/portfolios/all/portfolios/k1-8c-8s-v1-YEAR/`.
- *ek1*: `benchmarks/portfolios/eligible/portfolios/ek1-8c-8s-v2/ek1-8c-8s-v2-YEAR/`.

The new data is `benchmarks/portfolios/final-portfolios/portfolios-final/`
(used to build the AI training datasets). The median ratio is new / old per
instance — values >1 indicate the final-portfolios run was slower on that
shard. cpsat8 (vs standalone cp-sat) and k1 are systematically slower in the
final-portfolios run; ek1 is essentially unchanged. Combined with the
parasol-overhead replication results, the most likely explanation is uCloud
session-to-session machine variance: each (portfolio, shard) ran on a
different uCloud allocation, and some allocations were noticeably slower
than others.
"""

    OUT_FILE.write_text(typst)
    print(f"wrote {OUT_FILE}\n")
    print(f"{'portfolio':<14} {'year':<5} {'n':>4} {'old_med':>9} {'new_med':>9} {'med_ratio':>10}")
    for r in rows:
        print(f"{label[r['portfolio']]:<14} {r['year']:<5} {r['n']:>4} {fmt_ms(r['med_old']):>8}s {fmt_ms(r['med_new']):>8}s {r['med_ratio']:>9.2f}x")


if __name__ == "__main__":
    main()
