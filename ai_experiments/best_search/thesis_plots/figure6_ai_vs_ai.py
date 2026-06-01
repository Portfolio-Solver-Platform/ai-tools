#!/usr/bin/env python3
import csv
import os
import sys
from pathlib import Path

import numpy as np

for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env, "1")

HERE = Path(__file__).resolve().parent
BEST = HERE.parent
ROOT_AI = BEST.parents[1]
OOF_K1 = BEST / "oof_bagsvm_logstd.npz"
OOF_EK1 = BEST / "oof_bagsvm_logstd_ek1.npz"
DATA_3WAY = ROOT_AI / "data" / "portfolios_cpsat8_k1_ek1_training_data.npz"
OUT = HERE / "out"


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_text(path, content):
    Path(path).write_text(content)


def fmt_float_list(values, prec=2):
    return "(" + ", ".join(f"{v:.{prec}f}" for v in values) + ")"


def make_index(meta):
    return {(int(m["year"]), str(m["problem"]), str(m["model"]), str(m["name"])): i
            for i, m in enumerate(meta)}


def main():
    k1 = np.load(OOF_K1, allow_pickle=True)
    ek1 = np.load(OOF_EK1, allow_pickle=True)
    three = np.load(DATA_3WAY, allow_pickle=True)
    Y3 = three["Y"]
    meta3 = three["meta"]
    years3 = meta3["year"]

    sys.path.insert(0, str(ROOT_AI))
    from utils.shared_data import get_cpsat8_k1_data, get_cpsat8_ek1_data
    _, _, meta_k1 = get_cpsat8_k1_data()
    _, _, meta_ek1 = get_cpsat8_ek1_data()
    idx3 = make_index(meta3)
    idx_k1 = make_index(meta_k1)
    idx_ek1 = make_index(meta_ek1)

    common = []
    for key, i3 in idx3.items():
        if key in idx_k1 and key in idx_ek1:
            common.append((i3, idx_k1[key], idx_ek1[key]))
    print(f"aligned {len(common)} instances across all 3 datasets")

    Y3_aligned = np.array([Y3[i3] for i3, _, _ in common])
    pred_k1_aligned = np.array([k1["pred"][ik] for _, ik, _ in common])
    pred_ek1_aligned = np.array([ek1["pred"][ie] for _, _, ie in common])
    years_aligned = np.array([years3[i3] for i3, _, _ in common])

    ai_k1_borda = np.where(pred_k1_aligned == 0, Y3_aligned[:, 0], Y3_aligned[:, 1])
    ai_ek1_borda = np.where(pred_ek1_aligned == 0, Y3_aligned[:, 0], Y3_aligned[:, 2])

    always_cp = Y3_aligned[:, 0]
    always_k1 = Y3_aligned[:, 1]
    always_ek1 = Y3_aligned[:, 2]
    oracle = Y3_aligned.max(axis=1)

    unique_years = sorted(np.unique(years_aligned).tolist())
    per_year_rows = []
    for y in unique_years:
        m = years_aligned == y
        per_year_rows.append({
            "year": int(y),
            "n": int(m.sum()),
            "always_cpsat":   float(always_cp[m].sum()),
            "always_k1":      float(always_k1[m].sum()),
            "always_ek1":     float(always_ek1[m].sum()),
            "ai_k1":          float(ai_k1_borda[m].sum()),
            "ai_ek1":         float(ai_ek1_borda[m].sum()),
            "oracle":         float(oracle[m].sum()),
        })

    totals = {
        "always_cpsat": float(always_cp.sum()),
        "always_k1":    float(always_k1.sum()),
        "always_ek1":   float(always_ek1.sum()),
        "ai_k1":        float(ai_k1_borda.sum()),
        "ai_ek1":       float(ai_ek1_borda.sum()),
        "oracle":       float(oracle.sum()),
    }

    print("\nTotals across all years (3-way Borda):")
    for k, v in totals.items():
        ratio = v / totals["oracle"]
        print(f"  {k:14s}  {v:8.2f}  ratio={ratio:.3f}")

    head_to_head = {
        "n_total":        len(common),
        "ai_k1_wins":     int(np.sum(ai_k1_borda > ai_ek1_borda)),
        "ai_ek1_wins":    int(np.sum(ai_k1_borda < ai_ek1_borda)),
        "ties":           int(np.sum(ai_k1_borda == ai_ek1_borda)),
        "both_picked_cpsat": int(np.sum((pred_k1_aligned == 0) & (pred_ek1_aligned == 0))),
        "ai_k1_picked_alt":  int(np.sum(pred_k1_aligned == 1)),
        "ai_ek1_picked_alt": int(np.sum(pred_ek1_aligned == 1)),
        "disagree":          int(np.sum(pred_k1_aligned != pred_ek1_aligned)),
    }

    print("\nHead-to-head pairwise:")
    for k, v in head_to_head.items():
        print(f"  {k}: {v}")

    write_csv(OUT / "figure6_ai_vs_ai_per_year.csv",
              ["year", "n", "always_cpsat", "always_k1", "always_ek1",
               "ai_k1", "ai_ek1", "oracle"],
              [{k: f"{v:.2f}" if isinstance(v, float) else v for k, v in r.items()}
               for r in per_year_rows])

    write_csv(OUT / "figure6_ai_vs_ai_totals.csv",
              ["method", "borda", "ratio"],
              [{"method": k, "borda": f"{v:.2f}",
                "ratio": f"{v/totals['oracle']:.3f}"}
               for k, v in totals.items()])

    years_lst = [r["year"] for r in per_year_rows]
    cp_v = [r["always_cpsat"] for r in per_year_rows]
    aik1 = [r["ai_k1"] for r in per_year_rows]
    aiek1 = [r["ai_ek1"] for r in per_year_rows]
    or_v = [r["oracle"] for r in per_year_rows]

    n = len(years_lst)
    y_max = max(or_v) * 1.05
    y_min = min(cp_v) * 0.92
    tick_pairs = "(" + ", ".join(f"({i}, [{y}])" for i, y in enumerate(years_lst)) + ")"

    parts = [
        "#figure(",
        "  {",
        f"    let n = {n}",
        "    let xs = range(n).map(i => float(i))",
        "    lq.diagram(",
        "      width: 16cm, height: 7cm,",
        "      title: [Per-year 3-way Borda: AI-k1 vs AI-ek1 deployment],",
        "      xlabel: [Year], ylabel: [Borda (3-way tournament)],",
        f"      xaxis: (ticks: {tick_pairs}, subticks: none),",
        f"      ylim: ({y_min:.0f}, {y_max:.0f}),",
        f"      lq.bar(xs.map(x => x - 0.3), {fmt_float_list(cp_v)}, "
        "width: 0.2, fill: gray.lighten(30%), label: [always-cpsat]),",
        f"      lq.bar(xs.map(x => x - 0.1), {fmt_float_list(aik1)}, "
        "width: 0.2, fill: blue.lighten(20%), label: [AI-k1 deployment]),",
        f"      lq.bar(xs.map(x => x + 0.1), {fmt_float_list(aiek1)}, "
        "width: 0.2, fill: orange.lighten(20%), label: [AI-ek1 deployment]),",
        f"      lq.bar(xs.map(x => x + 0.3), {fmt_float_list(or_v)}, "
        "width: 0.2, fill: green.lighten(50%), label: [Oracle (best of 3)]),",
        "    )",
        "  },",
        "  caption: [Per-year LOYO 3-way Borda comparison of the two AI deployments. "
        "Each year is held out; both AIs are independently fit on the other 14 years and "
        "evaluated on the held-out year's instances. Each AI's pick (cpsat vs its alternative) "
        "is scored in a 3-way pairwise tournament against {cpsat, k1, ek1}, so the two "
        "deployment strategies are directly comparable on the same scale. always-cpsat and "
        "oracle (per-instance best of all 3) bound the achievable range.],",
        ") <fig:ai-vs-ai>",
        "",
    ]
    write_text(OUT / "figure6_ai_vs_ai.typ", "\n".join(parts))

    body = "#figure(\n  table(\n    columns: 3,\n"
    body += "    align: (left, right, right),\n"
    body += "    stroke: 0.5pt,\n"
    body += "    table-header([Method], [3-way Borda], [Ratio to Oracle]),\n"
    for k, v in totals.items():
        if k == "oracle":
            label = "Oracle (best of cpsat/k1/ek1)"
        elif k.startswith("always_"):
            label = "always-" + k[7:]
        else:
            label = "AI-" + k[3:] + " deployment"
        body += f"    [{label}], [{v:.2f}], [{v/totals['oracle']:.3f}],\n"
    body += "  ),\n"
    body += "  caption: [Total LOYO Borda across all 15 years, scored against the 3-way "
    body += "tournament of {cpsat, k1, ek1}. Each AI's prediction maps to a portfolio choice; "
    body += "that portfolio's 3-way Borda on the instance is added to the AI's total. The "
    body += "two AI deployments are evaluated on the same scale and are therefore directly "
    body += "comparable.],\n"
    body += ") <tab:ai-vs-ai-totals>\n"
    write_text(OUT / "figure6_ai_vs_ai_totals.typ", body)

    print(f"\nwrote outputs to {OUT}")


if __name__ == "__main__":
    main()
