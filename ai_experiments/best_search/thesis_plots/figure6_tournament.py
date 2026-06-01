#!/usr/bin/env python3
import csv
import json
import os
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
OOF_DIR = BEST / "oof"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT_AI))
sys.path.insert(0, str(BEST))

from utils.borda import _compare, _parse_obj, load_problem_types
from utils.cross_solver_eval import leave_one_year_out_folds

from preprocessing import SignedLog1p
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import Parallel, delayed


PROBLEM_TYPES_CSV = ROOT_AI / "benchmarks/open-category-benchmarks/problem_types.csv"
COMBINED_MEDIAN_CSV = ROOT_AI / "benchmarks/portfolios/final-portfolios/combined_median.csv"


def median_params_per_fold(folds_csv):
    raw = defaultdict(list)
    with open(folds_csv) as f:
        for r in csv.DictReader(f):
            if r["experiment"] == "BagSVM-MW/log_std":
                raw[r["fold_label"]].append(json.loads(r["best_params"]))
    out = {}
    for fold, lst in raw.items():
        out[fold] = {
            k: float(np.median([p[k] for p in lst]))
            for k in ("C", "gamma", "wpow", "max_samples")
        }
    return out


def fit_predict_fold(X_tr, y_tr, Yb_tr, X_te, params, n_estimators=10):
    pre = Pipeline([("log", SignedLog1p()), ("scaler", StandardScaler())])
    pre.fit(X_tr)
    Xs_tr = pre.transform(X_tr)
    Xs_te = pre.transform(X_te)
    w = np.abs(Yb_tr[:, 1] - Yb_tr[:, 0]) ** params["wpow"] + 1e-12
    n = len(X_tr)
    sample_n = int(params["max_samples"] * n)
    idx_pos = np.where(y_tr == 1)[0]
    idx_neg = np.where(y_tr == 0)[0]
    n_pos = int(round(sample_n * len(idx_pos) / n))
    n_neg = sample_n - n_pos

    probs = np.zeros((len(X_te), 2))
    for seed in range(n_estimators):
        rng = np.random.default_rng(seed)
        sel = np.concatenate([
            rng.choice(idx_pos, size=n_pos, replace=True),
            rng.choice(idx_neg, size=n_neg, replace=True),
        ])
        base = SVC(kernel="rbf", probability=True, random_state=seed,
                   C=params["C"], gamma=params["gamma"])
        base.fit(Xs_tr[sel], y_tr[sel], sample_weight=w[sel])
        probs += base.predict_proba(Xs_te)
    return probs / n_estimators


def get_or_generate_oof(stem, folds_csv_name, oof_filename):
    out_path = OOF_DIR / oof_filename
    if out_path.exists():
        return np.load(out_path, allow_pickle=True)

    npz_path = ROOT_AI / "data" / f"portfolios_{stem}_training_data.npz"
    d = np.load(npz_path, allow_pickle=True)
    X = d["X"]; Y = d["Y"]; meta = d["meta"]
    y_labels = np.argmax(Y, axis=1)
    years = meta["year"]

    params_per_fold = median_params_per_fold(RES / folds_csv_name)
    folds = leave_one_year_out_folds(years)

    def one_fold(fold_label, tr, te):
        params = params_per_fold[str(fold_label)]
        probs = fit_predict_fold(X[tr], y_labels[tr], Y[tr], X[te], params)
        return te, probs

    results = Parallel(n_jobs=15)(
        delayed(one_fold)(fl, tr, te) for fl, tr, te in folds
    )

    mean_p = np.zeros((len(X), 2))
    for te, probs in results:
        mean_p[te] = probs
    pred = np.argmax(mean_p, axis=1)

    OOF_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean_p=mean_p, pred=pred,
             y_labels=y_labels, Y_borda=Y, years=years, meta=meta)
    print(f"  saved {out_path}")
    return np.load(out_path, allow_pickle=True)


def load_combined_median():
    out = defaultdict(dict)
    with open(COMBINED_MEDIAN_CSV) as f:
        for r in csv.DictReader(f):
            key = (int(r["year"]), r["problem"], r["model"], r["name"])
            out[key][r["schedule"]] = r
    return out


def cmp_results(a, b, kind):
    return _compare(
        a["status"], float(a["time_ms"]), _parse_obj(a["objective"]),
        b["status"], float(b["time_ms"]), _parse_obj(b["objective"]),
        kind,
    )


def pick(prediction, cpsat_row, alt_row):
    return cpsat_row if int(prediction) == 0 else alt_row


def three_way_tournament():
    print("Generating / loading OOF for svc_k1 ...")
    oof_k1 = get_or_generate_oof(
        "cpsat8_k1", "folds.csv", "oof_BagSVM-MW__log_std.npz"
    )
    print("Generating / loading OOF for svc_ek1 ...")
    oof_ek1 = get_or_generate_oof(
        "cpsat8_ek1", "folds_ek1.csv", "oof_BagSVM-MW__log_std_ek1.npz"
    )

    pred_k1 = oof_k1["pred"]
    pred_ek1 = oof_ek1["pred"]

    train_k1 = np.load(ROOT_AI / "data/portfolios_cpsat8_k1_training_data.npz",
                       allow_pickle=True)
    train_ek1 = np.load(ROOT_AI / "data/portfolios_cpsat8_ek1_training_data.npz",
                        allow_pickle=True)
    meta_k1 = train_k1["meta"]
    meta_ek1 = train_ek1["meta"]
    assert len(meta_k1) == len(meta_ek1) == len(pred_k1) == len(pred_ek1)
    for i in range(len(meta_k1)):
        a = (int(meta_k1[i]["year"]), str(meta_k1[i]["problem"]),
             str(meta_k1[i]["model"]), str(meta_k1[i]["name"]))
        b = (int(meta_ek1[i]["year"]), str(meta_ek1[i]["problem"]),
             str(meta_ek1[i]["model"]), str(meta_ek1[i]["name"]))
        if a != b:
            raise RuntimeError(f"instance {i} mismatch: {a} vs {b}")

    n = len(meta_k1)
    instance_keys = [
        (int(meta_k1[i]["year"]), str(meta_k1[i]["problem"]),
         str(meta_k1[i]["model"]), str(meta_k1[i]["name"]))
        for i in range(n)
    ]

    per_inst = load_combined_median()
    problem_types = load_problem_types(PROBLEM_TYPES_CSV)

    score_cpsat = 0.0
    score_svc_k1 = 0.0
    score_svc_ek1 = 0.0
    score_oracle3 = 0.0
    h2h_cp_vs_k1 = 0.0
    h2h_k1_vs_cp = 0.0
    h2h_cp_vs_ek1 = 0.0
    h2h_ek1_vs_cp = 0.0
    aligned = 0
    per_year = defaultdict(lambda: {
        "cpsat": 0.0, "svc_k1": 0.0, "svc_ek1": 0.0, "oracle3": 0.0,
        "h2h_cp_vs_k1": 0.0, "h2h_k1_vs_cp": 0.0,
        "h2h_cp_vs_ek1": 0.0, "h2h_ek1_vs_cp": 0.0,
        "n": 0,
    })

    n_svc_k1_picks_k1 = 0
    n_svc_ek1_picks_ek1 = 0

    for i, key in enumerate(instance_keys):
        per_p = per_inst.get(key)
        if per_p is None:
            continue
        if "cpsat8" not in per_p or "k1-8c-8s-v1" not in per_p or "ek1-8c-8s-v2" not in per_p:
            continue

        cpsat_row = per_p["cpsat8"]
        k1_row = per_p["k1-8c-8s-v1"]
        ek1_row = per_p["ek1-8c-8s-v2"]
        kind = problem_types.get((key[1], key[2]), "SAT")

        sub_cpsat = cpsat_row
        sub_svc_k1 = pick(pred_k1[i], cpsat_row, k1_row)
        sub_svc_ek1 = pick(pred_ek1[i], cpsat_row, ek1_row)
        if int(pred_k1[i]) == 1:
            n_svc_k1_picks_k1 += 1
        if int(pred_ek1[i]) == 1:
            n_svc_ek1_picks_ek1 += 1

        s_cp_k1,   s_k1_cp   = cmp_results(sub_cpsat,  sub_svc_k1,  kind)
        s_cp_ek1,  s_ek1_cp  = cmp_results(sub_cpsat,  sub_svc_ek1, kind)
        s_k1_ek1,  s_ek1_k1  = cmp_results(sub_svc_k1, sub_svc_ek1, kind)

        cp_pts  = s_cp_k1 + s_cp_ek1
        k1_pts  = s_k1_cp + s_k1_ek1
        ek1_pts = s_ek1_cp + s_ek1_k1

        score_cpsat += cp_pts
        score_svc_k1 += k1_pts
        score_svc_ek1 += ek1_pts

        h2h_cp_vs_k1  += s_cp_k1
        h2h_k1_vs_cp  += s_k1_cp
        h2h_cp_vs_ek1 += s_cp_ek1
        h2h_ek1_vs_cp += s_ek1_cp

        a, b, c = cpsat_row, k1_row, ek1_row
        sab_a, sab_b = cmp_results(a, b, kind)
        sac_a, sac_c = cmp_results(a, c, kind)
        sbc_b, sbc_c = cmp_results(b, c, kind)
        oracle_pts = max(sab_a + sac_a, sab_b + sbc_b, sac_c + sbc_c)
        score_oracle3 += oracle_pts

        aligned += 1
        y = key[0]
        per_year[y]["cpsat"]   += cp_pts
        per_year[y]["svc_k1"]  += k1_pts
        per_year[y]["svc_ek1"] += ek1_pts
        per_year[y]["oracle3"] += oracle_pts
        per_year[y]["h2h_cp_vs_k1"]  += s_cp_k1
        per_year[y]["h2h_k1_vs_cp"]  += s_k1_cp
        per_year[y]["h2h_cp_vs_ek1"] += s_cp_ek1
        per_year[y]["h2h_ek1_vs_cp"] += s_ek1_cp
        per_year[y]["n"]       += 1

    max_per_instance = 2.0
    max_total = max_per_instance * aligned

    return {
        "n_instances": aligned,
        "max_total":   max_total,
        "totals": {
            "cpsat":   score_cpsat,
            "svc_k1":  score_svc_k1,
            "svc_ek1": score_svc_ek1,
            "oracle3": score_oracle3,
            "h2h_cp_vs_k1":  h2h_cp_vs_k1,
            "h2h_k1_vs_cp":  h2h_k1_vs_cp,
            "h2h_cp_vs_ek1": h2h_cp_vs_ek1,
            "h2h_ek1_vs_cp": h2h_ek1_vs_cp,
        },
        "per_year": dict(sorted(per_year.items())),
        "n_svc_k1_picks_k1":   n_svc_k1_picks_k1,
        "n_svc_ek1_picks_ek1": n_svc_ek1_picks_ek1,
    }


def emit_figure(res):
    totals = res["totals"]
    n = res["n_instances"]
    max_total = res["max_total"]

    csv_rows = [
        {"competitor": "cpsat (no AI)",
         "borda_3way": f"{totals['cpsat']:.2f}",
         "fraction_of_max": f"{totals['cpsat']/max_total:.4f}"},
        {"competitor": "svc_k1 (BagSVC-MW)",
         "borda_3way": f"{totals['svc_k1']:.2f}",
         "fraction_of_max": f"{totals['svc_k1']/max_total:.4f}"},
        {"competitor": "svc_ek1 (BagSVC-MW)",
         "borda_3way": f"{totals['svc_ek1']:.2f}",
         "fraction_of_max": f"{totals['svc_ek1']/max_total:.4f}"},
    ]
    with open(OUT / "figure6_three_way_tournament.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["competitor", "borda_3way", "fraction_of_max"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)

    py_rows = []
    py_rows.append({"year": "TOTAL", "n": n,
                    "cpsat":   f"{totals['cpsat']:.2f}",
                    "svc_k1":  f"{totals['svc_k1']:.2f}",
                    "svc_ek1": f"{totals['svc_ek1']:.2f}",
                    "oracle3": f"{totals['oracle3']:.2f}"})
    for y, d in res["per_year"].items():
        py_rows.append({"year": y, "n": d["n"],
                        "cpsat":   f"{d['cpsat']:.2f}",
                        "svc_k1":  f"{d['svc_k1']:.2f}",
                        "svc_ek1": f"{d['svc_ek1']:.2f}",
                        "oracle3": f"{d['oracle3']:.2f}"})
    with open(OUT / "figure6_three_way_per_year.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["year", "n", "cpsat", "svc_k1", "svc_ek1", "oracle3"]
        )
        w.writeheader()
        for r in py_rows:
            w.writerow(r)

    labels = ["cpsat", "svc_k1", "svc_ek1"]
    values = [totals["cpsat"], totals["svc_k1"], totals["svc_ek1"]]

    vmin_data = min(values)
    vmax_data = max(values)
    span = vmax_data - vmin_data
    ylo = vmin_data - 0.6 * span
    yhi = vmax_data + 0.3 * span

    tick_pairs = "(" + ", ".join(f"({i}, [{l}])" for i, l in enumerate(labels)) + ")"

    typ = []
    typ.append("#figure(")
    typ.append("  {")
    typ.append(f"    let n = {len(labels)}")
    typ.append("    let xs = range(n).map(i => float(i))")
    typ.append("    lq.diagram(")
    typ.append("      width: 11cm, height: 6cm,")
    typ.append("      title: [3-way LOYO Borda tournament],")
    typ.append("      xlabel: [Submission], ylabel: [Total Borda (out of "
               f"{max_total:.0f})],")
    typ.append(f"      xaxis: (ticks: {tick_pairs}, subticks: none),")
    typ.append(f"      ylim: ({ylo:.0f}, {yhi:.0f}),")
    typ.append("      lq.bar(xs, ("
               + ", ".join(f"{v:.2f}" for v in values)
               + ",), fill: blue.lighten(20%)),")
    typ.append("    )")
    typ.append("  },")
    typ.append("  caption: ["
               "Simulated MiniZinc Challenge among the three deployable submissions, "
               "scored by 3-way pairwise Borda using the MZN Challenge rules (see "
               "@minizinc_challenge). On every test instance each submission is "
               "compared head-to-head against the other two; each pairwise outcome "
               "lies in $[0, 1]$ (the better solution wins outright, equally-good "
               "solutions split the point in proportion to wall-clock time so the "
               "faster solver gets more). The per-instance score is the sum of the "
               f"two pairwise outcomes, and the maximum total per submission is "
               f"{max_total:.0f} = 2 #sym.times {n} instances; the y-axis is zoomed to "
               "make the gap between submissions readable. Predictions are "
               "leave-one-year-out: for each instance, the SVC's prediction comes from "
               "the model trained on the other 14 years."
               "],")
    typ.append("  )"
               " <fig:three-way-tournament>")
    typ.append("")
    res["_typ_3way"] = "\n".join(typ)

    print()
    print("=" * 70)
    print(f"3-WAY TOURNAMENT — {n} aligned instances, max per submission = {max_total:.0f}")
    print("=" * 70)
    for label, val in zip(labels + ["oracle (best-of-3)"],
                          values + [totals["oracle3"]]):
        pct_v = val / max_total * 100
        bar = "#" * int(pct_v / 2)
        print(f"  {label:30s}  {val:8.2f}  ({pct_v:5.2f}% of max)  {bar}")
    print()
    print(f"  svc_k1 picked k1   on {res['n_svc_k1_picks_k1']}/{n} instances "
          f"({res['n_svc_k1_picks_k1']/n*100:.1f}%)")
    print(f"  svc_ek1 picked ek1 on {res['n_svc_ek1_picks_ek1']}/{n} instances "
          f"({res['n_svc_ek1_picks_ek1']/n*100:.1f}%)")
    print()
    winner = labels[int(np.argmax(values[:3]))]
    print(f"  Winner among the 3 deployable submissions: {winner}")


def emit_head_to_head(res, ai_label, ai_total_key, cp_total_key, slug, anchor):
    totals = res["totals"]
    n = res["n_instances"]
    max_total = float(n)

    cp_val = totals[cp_total_key]
    ai_val = totals[ai_total_key]

    with open(OUT / f"{slug}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["competitor", "borda_2way", "fraction_of_max"])
        w.writeheader()
        w.writerow({"competitor": "cpsat (no AI)",
                    "borda_2way": f"{cp_val:.2f}",
                    "fraction_of_max": f"{cp_val/max_total:.4f}"})
        w.writerow({"competitor": ai_label,
                    "borda_2way": f"{ai_val:.2f}",
                    "fraction_of_max": f"{ai_val/max_total:.4f}"})

    labels = ["cpsat", ai_label]
    values = [cp_val, ai_val]
    vmin_data = min(values)
    vmax_data = max(values)
    span = max(vmax_data - vmin_data, 1.0)
    ylo = vmin_data - 0.6 * span
    yhi = vmax_data + 0.3 * span
    tick_pairs = "(" + ", ".join(f"({i}, [{l}])" for i, l in enumerate(labels)) + ")"

    typ = []
    typ.append("#figure(")
    typ.append("  {")
    typ.append(f"    let n = {len(labels)}")
    typ.append("    let xs = range(n).map(i => float(i))")
    typ.append("    lq.diagram(")
    typ.append("      width: 11cm, height: 6cm,")
    typ.append(f"      title: [Head-to-head: {ai_label} vs cpsat],")
    typ.append("      xlabel: [Submission], ylabel: [Total Borda (out of "
               f"{max_total:.0f})],")
    typ.append(f"      xaxis: (ticks: {tick_pairs}, subticks: none),")
    typ.append(f"      ylim: ({ylo:.0f}, {yhi:.0f}),")
    typ.append("      lq.bar(xs, ("
               + ", ".join(f"{v:.2f}" for v in values)
               + ",), fill: blue.lighten(20%)),")
    typ.append("    )")
    typ.append("  },")
    typ.append("  caption: ["
               f"Head-to-head MZN-Challenge Borda between {ai_label} and the "
               "always-cpsat baseline, scored under the MZN Challenge rules "
               "(see @minizinc_challenge). On every instance the two submissions "
               "are compared once: the better solution wins outright, equally-good "
               "solutions split the point in proportion to wall-clock time so the "
               f"faster solver gets more. The maximum total per submission is "
               f"{max_total:.0f} = 1 #sym.times {n} instances; the y-axis is "
               "zoomed to make the gap readable. Predictions are leave-one-year-out."
               "],")
    typ.append(f"  ) <{anchor}>")
    typ.append("")
    res.setdefault("_typ_h2h", []).append("\n".join(typ))

    print()
    print("=" * 70)
    print(f"HEAD-TO-HEAD — {ai_label} vs cpsat ({n} instances, max = {max_total:.0f})")
    print("=" * 70)
    for label, val in zip(labels, values):
        pct_v = val / max_total * 100
        bar = "#" * int(pct_v / 2)
        print(f"  {label:30s}  {val:8.2f}  ({pct_v:5.2f}% of max)  {bar}")
    winner = labels[int(np.argmax(values))]
    print(f"  Winner: {winner}")


def main():
    res = three_way_tournament()
    emit_figure(res)
    emit_head_to_head(res, "svc_k1",  "h2h_k1_vs_cp",  "h2h_cp_vs_k1",
                      "figure6b_h2h_svc_k1_vs_cpsat",  "fig:h2h-svc-k1-vs-cpsat")
    emit_head_to_head(res, "svc_ek1", "h2h_ek1_vs_cp", "h2h_cp_vs_ek1",
                      "figure6c_h2h_svc_ek1_vs_cpsat", "fig:h2h-svc-ek1-vs-cpsat")

    combined = "\n".join([res["_typ_3way"]] + res["_typ_h2h"])
    Path(OUT / "figure6_tournament.typ").write_text(combined)
    for stale in ("figure6_three_way_tournament.typ",
                  "figure6b_h2h_svc_k1_vs_cpsat.typ",
                  "figure6c_h2h_svc_ek1_vs_cpsat.typ"):
        p = OUT / stale
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    main()
