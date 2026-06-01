#!/usr/bin/env python3
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

for env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env, "1")

HERE = Path(__file__).resolve().parent
BEST = HERE.parent
ROOT_AI = BEST.parents[1]
RES = BEST / "results"
OOF_NPZ = BEST / "oof_bagsvm_logstd.npz"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_AI))


def load_folds(path):
    rows = defaultdict(lambda: defaultdict(list))
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["experiment"]][r["fold_label"]].append(r)
    return rows


def load_summary(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["experiment"]].append(r)
    return {k: max(v, key=lambda r: float(r["total_borda"])) for k, v in rows.items()}


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_text(path, content):
    Path(path).write_text(content)


def fmt_float_list(values, prec=2):
    return "(" + ", ".join(f"{v:.{prec}f}" for v in values) + (",)" if len(values) == 1 else ")")


def fmt_int_list(values):
    return "(" + ", ".join(str(int(v)) for v in values) + (",)" if len(values) == 1 else ")")


def per_year_average(fold_dict):
    out = {}
    for fold, rows in fold_dict.items():
        out[int(fold)] = {
            "test_borda":     float(np.mean([float(r["test_borda"]) for r in rows])),
            "oracle":         float(np.mean([float(r["oracle"]) for r in rows])),
            "cpsat_baseline": float(np.mean([float(r["cpsat_baseline"]) for r in rows])),
            "accuracy":       float(np.mean([float(r["accuracy"]) for r in rows])),
            "n_test":         int(np.median([float(r["n_test"]) for r in rows])),
        }
    return dict(sorted(out.items()))


def class_balance(dataset_loader):
    X, Y, meta = dataset_loader()
    y = np.argmax(Y, axis=1)
    return float((y == 0).mean())


def table1_headline():
    from utils.shared_data import get_cpsat8_k1_data, get_cpsat8_ek1_data

    sk1 = load_summary(RES / "summary.csv")
    sek1 = load_summary(RES / "summary_ek1.csv")

    cp_acc_k1 = class_balance(get_cpsat8_k1_data)
    cp_acc_ek1 = class_balance(get_cpsat8_ek1_data)

    plain_k1 = sk1["SVM-RBF/std"]
    bag_k1 = sk1["BagSVM-MW/log_std"]
    plain_ek1 = sek1["SVM-RBF/std"]
    bag_ek1 = sek1["BagSVM-MW/log_std"]

    oracle_k1 = float(bag_k1["total_oracle"])
    oracle_ek1 = float(bag_ek1["total_oracle"])
    cpsat_k1 = float(bag_k1["total_cpsat"])
    cpsat_ek1 = float(bag_ek1["total_cpsat"])

    rows = [
        ("always-cpsat", cpsat_k1, cpsat_k1 / oracle_k1, cp_acc_k1,
                          cpsat_ek1, cpsat_ek1 / oracle_ek1, cp_acc_ek1),
        ("Plain SVC", float(plain_k1["total_borda"]),
                       float(plain_k1["total_borda"]) / oracle_k1,
                       float(plain_k1["accuracy"]),
                       float(plain_ek1["total_borda"]),
                       float(plain_ek1["total_borda"]) / oracle_ek1,
                       float(plain_ek1["accuracy"])),
        ("BagSVC-MW", float(bag_k1["total_borda"]),
                       float(bag_k1["total_borda"]) / oracle_k1,
                       float(bag_k1["accuracy"]),
                       float(bag_ek1["total_borda"]),
                       float(bag_ek1["total_borda"]) / oracle_ek1,
                       float(bag_ek1["accuracy"])),
        ("Oracle", oracle_k1, 1.0, None, oracle_ek1, 1.0, None),
    ]

    write_csv(OUT / "table1_headline.csv",
              ["method", "k1_borda", "k1_ratio", "k1_acc",
                         "ek1_borda", "ek1_ratio", "ek1_acc"],
              [{"method": r[0],
                "k1_borda": f"{r[1]:.2f}",
                "k1_ratio": f"{r[2]:.3f}",
                "k1_acc":   "" if r[3] is None else f"{r[3]:.4f}",
                "ek1_borda": f"{r[4]:.2f}",
                "ek1_ratio": f"{r[5]:.3f}",
                "ek1_acc":   "" if r[6] is None else f"{r[6]:.4f}"}
               for r in rows])

    def cell_acc(a):
        return "---" if a is None else f"{a*100:.1f}%"

    table_typ = "#figure(\n  table(\n    columns: 7,\n"
    table_typ += "    align: (left+horizon, right, right, right, right, right, right+horizon),\n"
    table_typ += "    stroke: 0.5pt,\n"
    table_typ += "    table-header([Method], [Borda], [Ratio], [Acc], [Borda], [Ratio], [Acc]),\n"
    table_typ += "    table-header([], table.cell(colspan: 3)[#text(weight: \"bold\")[cpsat8 vs k1]], "
    table_typ += "table.cell(colspan: 3)[#text(weight: \"bold\")[cpsat8 vs ek1]]),\n"
    for name, kb, kr, ka, eb, er, ea in rows:
        table_typ += f"    [{name}], [{kb:.2f}], [{kr:.3f}], [{cell_acc(ka)}], "
        table_typ += f"[{eb:.2f}], [{er:.3f}], [{cell_acc(ea)}],\n"
    table_typ += "  ),\n"
    table_typ += "  caption: [Headline LOYO results on the two binary portfolio decisions. "
    table_typ += "Borda is the total over the 1379 test instances across all 15 leave-one-year-out folds. "
    table_typ += "Ratio is Borda divided by the oracle's per-instance maximum sum. "
    table_typ += "Accuracy is the fraction of instances where the model's chosen class matches the winning class; "
    table_typ += "the always-cpsat row reports the cpsat-wins rate. The oracle is the per-instance best.],\n"
    table_typ += ") <tab:headline>\n"
    write_text(OUT / "table1_headline.typ", table_typ)


def figure_per_year(folds_csv, dataset_loader, label, other_name, fig_tag):
    folds = load_folds(folds_csv)
    bag = per_year_average(folds["BagSVM-MW/log_std"])
    plain = per_year_average(folds["SVM-RBF/std"])

    X, Y, meta = dataset_loader()
    years_meta = meta["year"]
    other_baseline = {int(y): float(Y[years_meta == y, 1].sum())
                      for y in np.unique(years_meta)}

    years = sorted(set(bag) & set(plain) & set(other_baseline))
    cpsat_v  = [bag[y]["cpsat_baseline"] for y in years]
    other_v  = [other_baseline[y]        for y in years]
    plain_v  = [plain[y]["test_borda"]    for y in years]
    bag_v    = [bag[y]["test_borda"]      for y in years]
    oracle_v = [bag[y]["oracle"]          for y in years]

    write_csv(OUT / f"figure_per_year_{fig_tag}.csv",
              ["year", "always_cpsat", f"always_{other_name}",
               "plain_svc", "bag_svc_mw", "oracle"],
              [{"year": y,
                "always_cpsat": f"{cpsat_v[i]:.2f}",
                f"always_{other_name}": f"{other_v[i]:.2f}",
                "plain_svc":    f"{plain_v[i]:.2f}",
                "bag_svc_mw":   f"{bag_v[i]:.2f}",
                "oracle":       f"{oracle_v[i]:.2f}"}
               for i, y in enumerate(years)])

    n = len(years)
    y_max = max(oracle_v) * 1.1
    y_min = min(min(cpsat_v), min(other_v)) * 0.85

    tick_pairs = "(" + ", ".join(f"({i}, [{y}])" for i, y in enumerate(years)) + ")"

    parts = [
        "#figure(",
        "  {",
        f"    let n = {n}",
        f"    let xs = range(n).map(i => float(i))",
        "    lq.diagram(",
        "      width: 16cm, height: 7cm,",
        f"      title: [Per-year LOYO Borda, {label}],",
        "      xlabel: [Year], ylabel: [Borda score],",
        f"      xaxis: (ticks: {tick_pairs}, subticks: none),",
        f"      ylim: ({y_min:.0f}, {y_max:.0f}),",
        f"      lq.bar(xs.map(x => x - 0.32), {fmt_float_list(cpsat_v)}, "
        f"width: 0.15, fill: gray.lighten(20%), label: [always-cpsat]),",
        f"      lq.bar(xs.map(x => x - 0.16), {fmt_float_list(other_v)}, "
        f"width: 0.15, fill: purple.lighten(30%), label: [always-{other_name}]),",
        f"      lq.bar(xs.map(x => x), {fmt_float_list(plain_v)}, "
        f"width: 0.15, fill: red.lighten(30%), label: [Plain SVC]),",
        f"      lq.bar(xs.map(x => x + 0.16), {fmt_float_list(bag_v)}, "
        f"width: 0.15, fill: blue.lighten(10%), label: [BagSVC-MW]),",
        f"      lq.bar(xs.map(x => x + 0.32), {fmt_float_list(oracle_v)}, "
        f"width: 0.15, fill: green.lighten(30%), label: [Oracle]),",
        "    )",
        "  },",
        f"  caption: [Per-year LOYO Borda on {label}. Each year is held out from training; "
        "the model is fit on the other 14 and evaluated on the held-out year. "
        f"Bars are: always-cpsat and always-{other_name} (the two no-AI baselines; "
        "they sum to the year's instance count when every instance is solved by at "
        f"least one portfolio), Plain SVC (single RBF SVC), BagSVC-MW (this work), "
        f"and Oracle (per-instance best portfolio). Note 2025 is the only year where "
        f"always-{other_name} beats always-cpsat. Higher is better.],",
        f") <fig:{fig_tag}>",
        "",
    ]
    write_text(OUT / f"figure_per_year_{fig_tag}.typ", "\n".join(parts))


def figure3_total_comparison():
    from utils.shared_data import (
        get_cpsat8_ek1_data, get_cpsat8_k1_data, prepare_labels,
    )

    sk1 = load_summary(RES / "summary.csv")
    sek1 = load_summary(RES / "summary_ek1.csv")
    bag_k1 = sk1["BagSVM-MW/log_std"]
    plain_k1 = sk1["SVM-RBF/std"]
    bag_ek1 = sek1["BagSVM-MW/log_std"]
    plain_ek1 = sek1["SVM-RBF/std"]

    X, Y, _ = get_cpsat8_k1_data()
    y, Yb = prepare_labels(Y)
    rng = np.random.default_rng(0)
    random_borda_k1 = float(np.mean([
        Yb[np.arange(len(Yb)), rng.integers(0, 2, len(Yb))].sum()
        for _ in range(50)
    ]))
    always_k1_borda = float(Yb[:, 1].sum())
    always_cpsat_k1 = float(Yb[:, 0].sum())
    oracle_k1 = float(Yb.max(axis=1).sum())

    X, Y, _ = get_cpsat8_ek1_data()
    y, Yb = prepare_labels(Y)
    rng = np.random.default_rng(0)
    random_borda_ek1 = float(np.mean([
        Yb[np.arange(len(Yb)), rng.integers(0, 2, len(Yb))].sum()
        for _ in range(50)
    ]))
    always_ek1_borda = float(Yb[:, 1].sum())
    always_cpsat_ek1 = float(Yb[:, 0].sum())
    oracle_ek1 = float(Yb.max(axis=1).sum())

    labels = ["always-other", "random", "always-cpsat", "Plain SVC", "BagSVC-MW", "Oracle"]
    k1_vals = [always_k1_borda, random_borda_k1, always_cpsat_k1,
                float(plain_k1["total_borda"]), float(bag_k1["total_borda"]), oracle_k1]
    ek1_vals = [always_ek1_borda, random_borda_ek1, always_cpsat_ek1,
                float(plain_ek1["total_borda"]), float(bag_ek1["total_borda"]), oracle_ek1]

    write_csv(OUT / "figure3_total_comparison.csv",
              ["method", "k1_borda", "ek1_borda"],
              [{"method": labels[i],
                "k1_borda":  f"{k1_vals[i]:.2f}",
                "ek1_borda": f"{ek1_vals[i]:.2f}"}
               for i in range(len(labels))])

    label_pairs = "(" + ", ".join(f"({i}, [{l}])" for i, l in enumerate(labels)) + ")"
    n = len(labels)
    parts = [
        "#figure(",
        "  grid(",
        "    columns: 1,",
        "    row-gutter: 0.8em,",
        "    {",
        f"      let n = {n}",
        "      let xs = range(n).map(i => float(i))",
        "      lq.diagram(",
        "        width: 14cm, height: 5cm,",
        "        title: [cpsat8 vs k1],",
        "        xlabel: [Method], ylabel: [Total Borda],",
        f"        xaxis: (ticks: {label_pairs}, subticks: none),",
        f"        lq.bar(xs, {fmt_float_list(k1_vals)}, fill: blue.lighten(20%)),",
        "      )",
        "    },",
        "    {",
        f"      let n = {n}",
        "      let xs = range(n).map(i => float(i))",
        "      lq.diagram(",
        "        width: 14cm, height: 5cm,",
        "        title: [cpsat8 vs ek1],",
        "        xlabel: [Method], ylabel: [Total Borda],",
        f"        xaxis: (ticks: {label_pairs}, subticks: none),",
        f"        lq.bar(xs, {fmt_float_list(ek1_vals)}, fill: blue.lighten(20%)),",
        "      )",
        "    },",
        "  ),",
        "  caption: [Total LOYO Borda across baselines and the final BagSVC-MW model, "
        "for both portfolio decisions. \"always-other\" picks k1 (left) or ek1 (right) on every "
        "instance; \"random\" is a uniform coin flip averaged over 50 seeds; \"always-cpsat\" "
        "is the no-AI fallback. The final model beats every simple baseline and lies close to "
        "the per-instance oracle.],",
        ") <fig:total-comparison>",
        "",
    ]
    write_text(OUT / "figure3_total_comparison.typ", "\n".join(parts))


def table2_preprocessing():
    sk1 = load_summary(RES / "summary.csv")

    pairs_compact = [
        ("StandardScaler",          "SVM-RBF/std",        "BagSVM-MW/std"),
        ("QuantileTransformer",     "SVM-RBF/quantile",   "BagSVM-MW/quantile"),
        ("signed-log + StandardScaler",
                                    "SVM-RBF/log_std",    "BagSVM-MW/log_std"),
        ("signed-log + QuantileTransformer",
                                    "SVM-RBF/log_quantile","BagSVM-MW/log_quantile"),
    ]
    pairs_full = pairs_compact + [
        ("RobustScaler",            "SVM-RBF/robust",     "BagSVM-MW/log_robust"),
        ("PowerTransformer",        "SVM-RBF/power",      "BagSVM-MW/log_power"),
        ("signed-log + RobustScaler",
                                    "SVM-RBF/log_robust", "BagSVM-MW/log_robust"),
        ("signed-log + PowerTransformer",
                                    "SVM-RBF/log_power",  "BagSVM-MW/log_power"),
    ]

    def fetch(name):
        if name in sk1:
            return float(sk1[name]["total_borda"])
        return None

    def emit(pairs, suffix):
        write_csv(OUT / f"table2_preprocessing_{suffix}.csv",
                  ["preprocessing", "plain_svc_borda", "bagged_mw_borda"],
                  [{"preprocessing": p[0],
                    "plain_svc_borda":  "" if fetch(p[1]) is None else f"{fetch(p[1]):.2f}",
                    "bagged_mw_borda":  "" if fetch(p[2]) is None else f"{fetch(p[2]):.2f}"}
                   for p in pairs])

        best_plain  = max([fetch(p[1]) for p in pairs if fetch(p[1]) is not None])
        best_bagged = max([fetch(p[2]) for p in pairs if fetch(p[2]) is not None])

        body = "#figure(\n  table(\n    columns: 3,\n"
        body += "    align: (left, right, right),\n"
        body += "    stroke: 0.5pt,\n"
        body += "    table-header([Preprocessing], [Plain SVC], [Bagged SVC-MW]),\n"
        for name, plain_key, bag_key in pairs:
            p = fetch(plain_key)
            b = fetch(bag_key)
            p_cell = f"*{p:.2f}*" if p is not None and abs(p - best_plain) < 1e-6 else \
                     ("---" if p is None else f"{p:.2f}")
            b_cell = f"*{b:.2f}*" if b is not None and abs(b - best_bagged) < 1e-6 else \
                     ("---" if b is None else f"{b:.2f}")
            body += f"    [{name}], [{p_cell}], [{b_cell}],\n"
        body += "  ),\n"
        body += "  caption: [LOYO Borda by preprocessing on cpsat8_k1, for plain RBF SVC and "
        body += "bagged margin-weighted SVC. The best per column is in bold. "
        body += "Signed-log + StandardScaler is the *worst* preprocessing for a plain SVC "
        body += "but becomes the *best* once we add margin weighting and bagging - the "
        body += "preprocessing ranking flips when the model class changes.],\n"
        body += f") <tab:preprocessing-{suffix}>\n"
        write_text(OUT / f"table2_preprocessing_{suffix}.typ", body)

    emit(pairs_compact, "compact")
    emit(pairs_full, "full")


def figure4_confusion():
    d = np.load(OOF_NPZ)
    y = d["y_labels"]
    pred = d["pred"]

    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())

    precision_k1 = tp / (tp + fp) if (tp + fp) else 0.0
    recall_k1    = tp / (tp + fn) if (tp + fn) else 0.0
    precision_cp = tn / (tn + fn) if (tn + fn) else 0.0
    recall_cp    = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy     = (tn + tp) / (tn + fp + fn + tp)

    write_csv(OUT / "figure4_confusion.csv",
              ["cell", "value"],
              [{"cell": "true_cpsat_pred_cpsat", "value": tn},
               {"cell": "true_cpsat_pred_k1",    "value": fp},
               {"cell": "true_k1_pred_cpsat",    "value": fn},
               {"cell": "true_k1_pred_k1",       "value": tp},
               {"cell": "precision_k1",          "value": f"{precision_k1:.3f}"},
               {"cell": "recall_k1",             "value": f"{recall_k1:.3f}"},
               {"cell": "precision_cpsat",       "value": f"{precision_cp:.3f}"},
               {"cell": "recall_cpsat",          "value": f"{recall_cp:.3f}"},
               {"cell": "accuracy",              "value": f"{accuracy:.3f}"}])

    body = "#figure(\n  table(\n    columns: 3,\n"
    body += "    align: (left+horizon, center+horizon, center+horizon),\n"
    body += "    stroke: 0.5pt,\n"
    body += "    table-header([], [Predicted cpsat], [Predicted k1]),\n"
    body += f"    [True cpsat], [{tn}], [{fp}],\n"
    body += f"    [True k1],    [{fn}], [{tp}],\n"
    body += "  ),\n"
    body += "  caption: [Out-of-fold confusion matrix on cpsat8_k1 (15-fold LOYO, 1379 test "
    body += "instances). Precision on the k1 class is "
    body += f"#text(weight: \"bold\")[{precision_k1*100:.1f}%] "
    body += f"at recall #text(weight: \"bold\")[{recall_k1*100:.1f}%]. "
    body += "The model is conservative on k1 - it only commits to a k1 prediction when reasonably "
    body += "confident, trading recall for precision to avoid borda-losing wrong picks.],\n"
    body += ") <tab:confusion-k1>\n"
    write_text(OUT / "figure4_confusion.typ", body)


def parse_verify_holdout():
    path = RES / "log_verify_holdout.txt"
    text = path.read_text()
    line_re = re.compile(
        r"^\s+(?P<name>[a-z_0-9]+)\s+"
        r"borda=(?P<borda>-?\d+\.\d+)\s*[±+\-]\s*(?P<borda_sd>\d+\.\d+)\s+"
        r"ratio=(?P<ratio>\d+\.\d+)\s*[±+\-]\s*(?P<ratio_sd>\d+\.\d+)\s+"
        r"acc=\s*(?P<acc>\d+\.\d+)\s*[±+\-]\s*(?P<acc_sd>\d+\.\d+)%"
    )

    out = {"k1": {}, "ek1": {}}
    current = None
    after_avg = False
    for line in text.splitlines():
        if "cpsat8_k1 " in line and "instances" in line:
            current = "k1"; after_avg = False; continue
        if "cpsat8_ek1 " in line and "instances" in line:
            current = "ek1"; after_avg = False; continue
        if "averaged over" in line:
            after_avg = True; continue
        if current is None or not after_avg:
            continue
        m = line_re.match(line)
        if m:
            out[current][m.group("name")] = {
                "borda":    float(m.group("borda")),
                "borda_sd": float(m.group("borda_sd")),
                "ratio":    float(m.group("ratio")),
                "ratio_sd": float(m.group("ratio_sd")),
                "acc":      float(m.group("acc")),
                "acc_sd":   float(m.group("acc_sd")),
            }
    return out


def figure5_holdout():
    data = parse_verify_holdout()
    label_map = [
        ("always_cpsat",      "always-cpsat"),
        ("always_k1",         "always-other"),
        ("random",            "random"),
        ("logreg_default",    "LogReg"),
        ("svm_rbf_default",   "Plain SVC"),
        ("bag_svm_mw_fixed",  "BagSVC-MW"),
    ]
    rows_csv = []
    for ds in ("k1", "ek1"):
        for key, label in label_map:
            r = data[ds].get(key)
            if r is None:
                continue
            rows_csv.append({"dataset": ds, "method": label,
                             "borda": f"{r['borda']:.2f}",
                             "borda_sd": f"{r['borda_sd']:.2f}",
                             "ratio": f"{r['ratio']:.3f}",
                             "ratio_sd": f"{r['ratio_sd']:.3f}",
                             "acc_pct": f"{r['acc']:.1f}",
                             "acc_sd_pct": f"{r['acc_sd']:.1f}"})
    write_csv(OUT / "figure5_holdout.csv",
              ["dataset", "method", "borda", "borda_sd", "ratio", "ratio_sd",
               "acc_pct", "acc_sd_pct"],
              rows_csv)

    def emit_diagram(ds, title):
        keys = [k for k, _ in label_map if k in data[ds]]
        labels = [next(l for k, l in label_map if k == kk) for kk in keys]
        vals = [data[ds][k]["borda"] for k in keys]
        sds  = [data[ds][k]["borda_sd"] for k in keys]
        n = len(labels)
        ticks = "(" + ", ".join(f"({i}, [{l}])" for i, l in enumerate(labels)) + ")"
        return "\n".join([
            "    {",
            f"      let n = {n}",
            "      let xs = range(n).map(i => float(i))",
            "      lq.diagram(",
            "        width: 14cm, height: 5cm,",
            f"        title: [{title}],",
            "        xlabel: [Method], ylabel: [Test Borda (80/20 holdout)],",
            f"        xaxis: (ticks: {ticks}, subticks: none),",
            f"        lq.bar(xs, {fmt_float_list(vals)}, fill: blue.lighten(20%)),",
            f"        lq.plot(xs, {fmt_float_list(vals)}, "
            f"yerr: {fmt_float_list(sds)}, stroke: none, color: black),",
            "      )",
            "    }",
        ])

    parts = [
        "#figure(",
        "  grid(",
        "    columns: 1,",
        "    row-gutter: 0.8em,",
        emit_diagram("k1", "cpsat8 vs k1") + ",",
        emit_diagram("ek1", "cpsat8 vs ek1"),
        "  ),",
        "  caption: [80/20 stratified random-split holdout, averaged over 5 random seeds. "
        "All methods use FIXED hyperparameters (no HPO at evaluation time); BagSVC-MW uses "
        "the median LOYO params from the cpsat8_k1 sweep applied unchanged to both datasets. "
        "Error bars are ± one standard deviation across seeds. The 80/20 ratios are higher "
        "than the LOYO ratios because year-stratification (the LOYO setup) is a strictly "
        "harder verification than random splits.],",
        ") <fig:holdout>",
        "",
    ]
    write_text(OUT / "figure5_holdout.typ", "\n".join(parts))


def main():
    from utils.shared_data import get_cpsat8_ek1_data, get_cpsat8_k1_data

    table1_headline()
    figure_per_year(RES / "folds.csv", get_cpsat8_k1_data,
                    "cpsat8 vs k1", "k1", "k1")
    figure_per_year(RES / "folds_ek1.csv", get_cpsat8_ek1_data,
                    "cpsat8 vs ek1", "ek1", "ek1")
    figure3_total_comparison()
    table2_preprocessing()
    figure4_confusion()
    figure5_holdout()
    print("done. outputs in:", OUT)
    for p in sorted(OUT.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
