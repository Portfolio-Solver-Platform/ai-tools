"""
Compute every reference / oracle / search number from the portfolio-selector
chapter of the report under a single consistent Borda convention:

    Convention #2: every candidate is scored against the 15 open-category
    ladder configs. When the candidate IS itself one of the 15 ladder
    configs, the self-comparison ties at 0.5 on solved instances and 0 on
    unsolved ones.

This is the convention that makes the cp-sat 8c reference (3693.71)
consistent with the k>=2 search outputs.

Implementation: utils.borda.borda_scores returns the no-self matrix S.
Under convention #2 the per-(config, instance) score is

    S[c, i] + 0.5 * is_ladder[c] * is_solved[c, i]

Everything downstream (oracles, the best-k search) is just per-instance
max over a config or member subset, then summed.

Usage:
    python verify_convention_v2.py            # both tracks
    python verify_convention_v2.py --skip-k3  # skip the slow k=3 ineligible run
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.borda import borda_scores, load_problem_types  # noqa: E402

OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
BKP_PATH = Path(__file__).parent / "best-k-portfolios.py"

INELIGIBLE_SOLVERS = {"solutions.huub", "org.chuffed.chuffed", "org.gecode.gecode"}
SOLVED = {"Optimal", "Satisfied", "Unsat"}


def load_data():
    rows = list(csv.DictReader(open(OPEN_CSV)))
    problem_types = load_problem_types(TYPES_CSV)
    open_cat = {(r["solver"], int(r["cores"])) for r in rows
                if r["open_category"] == "True"}
    scores, configs, instances = borda_scores(rows, problem_types,
                                              opponents=open_cat)
    config_idx = {c: i for i, c in enumerate(configs)}
    instance_idx = {k: i for i, k in enumerate(instances)}

    n_c, n_i = len(configs), len(instances)
    solved = np.zeros((n_c, n_i), dtype=bool)
    wrong = np.zeros((n_c, n_i), dtype=bool)
    for r in rows:
        ci = config_idx[(r["solver"], int(r["cores"]))]
        ii = instance_idx[(r["problem"], r["name"])]
        solved[ci, ii] = r["status"] in SOLVED
        if r.get("wrong") == "True":
            wrong[ci, ii] = True

    is_ladder = np.array([(c in open_cat) for c in configs], dtype=bool)
    scores_v2 = scores + 0.5 * is_ladder[:, None] * solved

    return {
        "rows": rows, "configs": configs, "instances": instances,
        "config_idx": config_idx, "open_cat": open_cat,
        "scores_v2": scores_v2, "wrong": wrong,
    }


def oracle(scores_v2, config_idx, pool):
    idxs = np.array([config_idx[c] for c in pool], dtype=int)
    return float(scores_v2[idxs].max(axis=0).sum())


def import_bkp():
    spec = importlib.util.spec_from_file_location("bkp", BKP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def search(scores_v2, configs, config_idx, wrong, excluded, ks, label):
    """Run the best-k search using scores_v2 in place of the no-self matrix.
    Returns {k: top_score}.
    """
    bkp = import_bkp()

    cfgs = bkp.prune_configs(scores_v2, configs)
    cfgs = [c for c in cfgs if c[0] not in excluded]

    portfolios = bkp.generate_portfolios(cfgs, config_idx, scores_v2, wrong=wrong)

    fixed = [("cp-sat", 8)]
    fixed_vec = scores_v2[[config_idx[c] for c in fixed]].max(axis=0)
    baseline = fixed_vec
    portfolios = [(v, p) for v, p in portfolios if p != fixed]
    portfolios = bkp.prune_portfolios(portfolios, baseline=baseline)

    out = {1: float(fixed_vec.sum())}
    for k_extra in (k - 1 for k in ks if k >= 2):
        res = bkp.best_k_portfolios(portfolios, k=k_extra,
                                    baseline=baseline, top_n=1)
        if res:
            s, _, _ = res[0]
            out[k_extra + 1] = float(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-k3", action="store_true",
                    help="skip k=3 ineligible (slow, ~5 min)")
    args = ap.parse_args()

    d = load_data()
    elig_pool = [c for c in d["configs"]
                 if c[0] not in INELIGIBLE_SOLVERS]
    elig_ladder = [c for c in d["open_cat"]
                   if c[0] not in INELIGIBLE_SOLVERS]
    cps_8c = ("cp-sat", 8)

    print("=" * 64)
    print("Sanity checks")
    print("=" * 64)
    cps = d["scores_v2"][d["config_idx"][cps_8c]].sum()
    print(f"  cp-sat 8c convention-#2 baseline: {cps:>8.2f}   (expect 3693.71)")
    print()

    print("=" * 64)
    print("Oracles under convention #2")
    print("=" * 64)
    pools = [
        ("Best-variant ineligible (15 ladder)",        sorted(d["open_cat"])),
        ("Best-variant eligible   (12 = 15 - h/c/g)",  sorted(elig_ladder)),
        ("All-config ineligible   (40 configs)",       list(d["configs"])),
        ("All-config eligible     (34 configs)",       elig_pool),
        ("Anchor test (14 non-cps ladder + cps 1c)",
         [c for c in d["open_cat"] if c[0] != "cp-sat"] + [("cp-sat", 1)]),
    ]
    for name, pool in pools:
        s = oracle(d["scores_v2"], d["config_idx"], pool)
        print(f"  {name:<48} {s:>8.2f}")
    print()

    print("=" * 64)
    print("Search under convention #2 (anchor cp-sat 8c, force cp-sat 1c)")
    print("=" * 64)
    elig_ks = [1, 2, 3]
    inel_ks = [1, 2] if args.skip_k3 else [1, 2, 3]

    inel = search(d["scores_v2"], d["configs"], d["config_idx"], d["wrong"],
                  excluded=set(), ks=inel_ks, label="ineligible")
    elig = search(d["scores_v2"], d["configs"], d["config_idx"], d["wrong"],
                  excluded=INELIGIBLE_SOLVERS, ks=elig_ks, label="eligible")

    print()
    print(f"  {'Track':<14} {'k':>3} {'top score':>10}")
    print(f"  {'-'*14} {'-'*3} {'-'*10}")
    for k in inel_ks:
        print(f"  {'Ineligible':<14} {k:>3} {inel.get(k, float('nan')):>10.2f}")
    for k in elig_ks:
        print(f"  {'Eligible':<14} {k:>3} {elig.get(k, float('nan')):>10.2f}")
    print()


if __name__ == "__main__":
    main()
