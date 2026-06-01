#!/usr/bin/env python3
"""Borda restricted to instances where the AI chose the alt portfolio.

For every (year, problem, name) where the AI picked class 1 (k1 or ek1)
rather than cpsat, we compare the AI's run against the cpsat8 baseline
using utils.borda._compare. Symmetric counterpart of
borda_when_ai_picks_cpsat.py: shows how much Borda the AI's decision
*earned* by deviating from cpsat (or *cost*, when it deviated wrong).
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import csv
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent.parent))
sys.path.insert(0, "/home/sofus/speciale/ai/parasol/command-line-ai")
from utils.borda import _compare, _parse_obj  # noqa: E402
from svc_common import BagSVCPredictor, SignedLog1p  # noqa: F401, E402

AI_MEDIAN_CSV    = ROOT / "combined_median.csv"
CPSAT_MEDIAN_CSV = ROOT.parent / "final-portfolios" / "combined_median.csv"
TYPES_CSV        = ROOT.parent.parent / "open-category-benchmarks" / "problem_types.csv"
MODELS_DIR       = Path("/home/sofus/speciale/ai/ai-tools/ai_experiments/pre2025/models")
DATA_DIR         = Path("/home/sofus/speciale/ai/ai-tools/data")
OUT_CSV          = ROOT / "leaderboard_when_alt.csv"
OUT_TYP          = ROOT / "leaderboard_when_alt.typ"
OUT_INSTANCES    = ROOT / "alt_predicted_instances.csv"
PICK_CLASS       = 1
PICK_NAME        = "alt"

YEARS = ("2020", "2021", "2022")
AIS   = (("svc-k1", "k1"), ("svc-ek1", "ek1"))
MAX_TIME_MS = 1_200_000


def load_problem_types():
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(TYPES_CSV))}


def load_median(path, schedule_filter):
    out, models = {}, {}
    for r in csv.DictReader(open(path)):
        if r["schedule"] not in schedule_filter or r["year"] not in YEARS:
            continue
        key = (r["schedule"], r["year"], r["problem"], r["name"])
        out[key] = {
            "status":    r["status"] or "",
            "time_ms":   float(r["time_ms"]) if r["time_ms"] not in ("", None) else MAX_TIME_MS,
            "objective": _parse_obj(r["objective"]) if r["objective"] not in ("", None) else None,
        }
        models[(r["year"], r["problem"], r["name"])] = r["model"]
    return out, models


def predict_cpsat_mask(family, year):
    tag = "cpsat8_k1" if family == "k1" else "cpsat8_ek1"
    d = np.load(DATA_DIR / f"portfolios_{tag}_training_data.npz", allow_pickle=True)
    meta, X = d["meta"], d["X"]
    mask = np.array([int(m["year"]) == int(year) for m in meta])
    if not mask.any():
        return {}
    model = joblib.load(MODELS_DIR / f"svc_{family}_no{year}.joblib")
    preds = model.predict(X[mask]).astype(int)
    out = {}
    for m, pred in zip(meta[mask], preds):
        out[(str(m["problem"]), str(m["model"]), str(m["name"]))] = int(pred)
    return out


def main():
    types = load_problem_types()
    ai_rows,    ai_model    = load_median(AI_MEDIAN_CSV,    {"svc-k1", "svc-ek1"})
    cpsat_rows, cpsat_model = load_median(CPSAT_MEDIAN_CSV, {"cpsat8"})

    predicted = {}
    for sched, family in AIS:
        for year in YEARS:
            predicted[(sched, year)] = predict_cpsat_mask(family, year)
            n_alt = sum(1 for v in predicted[(sched, year)].values() if v == PICK_CLASS)
            n_total = len(predicted[(sched, year)])
            print(f"{sched} {year}: predicted {PICK_NAME} on {n_alt}/{n_total} "
                  f"({n_alt/max(n_total,1)*100:.1f}%) of features-available instances")
    print()

    inst_rows = []
    rows_out = []
    print(f"{'ai':<10} {'year':<6} {'n_'+PICK_NAME+'_pick':>13} "
          f"{'borda_ai':>9} {'borda_cpsat':>12} {'norm':>6}")

    for sched, family in AIS:
        ai_running, cp_running, n_running = 0.0, 0.0, 0
        for year in YEARS:
            ai_total, cp_total, n_used = 0.0, 0.0, 0
            for (sched_k, y, p, n), ai_r in ai_rows.items():
                if sched_k != sched or y != year:
                    continue
                cp = cpsat_rows.get(("cpsat8", year, p, n))
                if cp is None:
                    continue
                model = ai_model.get((year, p, n)) or cpsat_model.get((year, p, n))
                if model is None:
                    continue
                pred = predicted[(sched, year)].get((p, model, n))
                if pred is None or pred != PICK_CLASS:
                    continue
                kind = types.get((p, model))
                if kind is None:
                    continue
                sa, sb = _compare(
                    ai_r["status"], ai_r["time_ms"], ai_r["objective"],
                    cp["status"],   cp["time_ms"],   cp["objective"],
                    kind,
                )
                ai_total += sa
                cp_total += sb
                n_used   += 1
                inst_rows.append({
                    "ai":          sched,
                    "year":        year,
                    "problem":     p,
                    "model":       model,
                    "name":        n,
                    "ai_status":   ai_r["status"],
                    "ai_time_ms":  int(ai_r["time_ms"]),
                    "ai_obj":      ai_r["objective"] if ai_r["objective"] is not None else "",
                    "cp_status":   cp["status"],
                    "cp_time_ms":  int(cp["time_ms"]),
                    "cp_obj":      cp["objective"] if cp["objective"] is not None else "",
                    "borda_ai":    round(sa, 4),
                    "borda_cpsat": round(sb, 4),
                })
            norm = ai_total / n_used if n_used else 0.0
            rows_out.append({
                "ai":           sched,
                "year":         year,
                "n_pick": n_used,
                "borda_ai":     round(ai_total, 2),
                "borda_cpsat":  round(cp_total, 2),
                "normalized":   round(norm, 4),
            })
            ai_running += ai_total
            cp_running += cp_total
            n_running  += n_used
            print(f"{sched:<10} {year:<6} {n_used:>13} {ai_total:>9.2f} "
                  f"{cp_total:>12.2f} {norm:>6.4f}")

        norm = ai_running / n_running if n_running else 0.0
        rows_out.append({
            "ai":           sched,
            "year":         "TOTAL",
            "n_pick": n_running,
            "borda_ai":     round(ai_running, 2),
            "borda_cpsat":  round(cp_running, 2),
            "normalized":   round(norm, 4),
        })
        print(f"{sched:<10} {'TOTAL':<6} {n_running:>13} {ai_running:>9.2f} "
              f"{cp_running:>12.2f} {norm:>6.4f}")
        print()

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ai", "year", "n_pick", "borda_ai", "borda_cpsat", "normalized",
        ])
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote -> {OUT_CSV.relative_to(ROOT)}")

    with open(OUT_INSTANCES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ai", "year", "problem", "model", "name",
            "ai_status", "ai_time_ms", "ai_obj",
            "cp_status", "cp_time_ms", "cp_obj",
            "borda_ai", "borda_cpsat",
        ])
        w.writeheader()
        w.writerows(inst_rows)
    print(f"Wrote -> {OUT_INSTANCES.relative_to(ROOT)}")

    typ = []
    typ.append("#figure(")
    typ.append("  table(")
    typ.append("    columns: 6,")
    typ.append("    align: (left, right, right, right, right, right),")
    typ.append("    table.header(")
    typ.append("      [AI], [Year], [n alt-picked], "
               "[Borda AI], [Borda cpsat8], [Norm.],")
    typ.append("    ),")
    for i, r in enumerate(rows_out):
        if i > 0 and r["ai"] != rows_out[i - 1]["ai"]:
            typ.append("    table.hline(),")
        year_label = r["year"] if r["year"] != "TOTAL" else "*total*"
        typ.append(
            f"    [{r['ai']}], [{year_label}], [{r['n_pick']}], "
            f"[{r['borda_ai']:.2f}], [{r['borda_cpsat']:.2f}], "
            f"[{r['normalized']:.4f}],"
        )
    typ.append("  ),")
    typ.append("  caption: ["
               "Head-to-head Borda restricted to instances where the AI "
               "*deviated* from cpsat by picking the alt portfolio "
               "(k1-8c-8s-v1 for svc-k1, ek1-8c-8s-v2 for svc-ek1). "
               "These rows measure the *decision quality* of the AI's "
               "non-cpsat picks: a normalised score above 0.5 means the "
               "alt portfolio beat cpsat on the instances the AI chose to "
               "deviate on, below 0.5 means the deviation cost Borda."
               "],")
    typ.append("  ) <tab:vs-cpsat-when-alt>")
    typ.append("")
    OUT_TYP.write_text("\n".join(typ))
    print(f"Wrote -> {OUT_TYP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
