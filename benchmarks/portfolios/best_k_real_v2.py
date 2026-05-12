"""
Best-k combinations of *actually-run* portfolios under convention #2.

Mirrors best-k-portfolios.py but the candidates are the portfolios that were
actually run (rows in benchmarks/portfolios/{all,eligible}/combined.csv),
not synthesised k-of-config tuples. cp-sat(8c) is anchored as in the
synthesis search; its per-instance score includes the +0.5 self-tie when
solved (convention #2). Portfolios are scored against the 15 open-cat ladder
the usual way (no self-tie applies, since they aren't in the ladder).

Reuses prune_portfolios and best_k_portfolios from best-k-portfolios.py.

Usage:
    python best_k_real_v2.py 1 --track ineligible   # k=2 ineligible
    python best_k_real_v2.py 2 --track eligible     # k=3 eligible
"""
import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.borda import _compare, _parse_obj  # noqa: E402

OPEN_CSV   = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
TYPES_CSV  = ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
ALL_CSV    = ROOT / "benchmarks/portfolios/all/combined.csv"
ELIG_CSV   = ROOT / "benchmarks/portfolios/eligible/combined.csv"
BKP_PATH   = ROOT / "find-best-portfolio-experiment/on-own-data/best-k-portfolios.py"

CPSAT_8C = ("cp-sat", 8)
MAX_TIME_MS = 1_200_000


def load_problem_types(path):
    return {(r["problem"], r["model"]): r["type"]
            for r in csv.DictReader(open(path))}


def make_row(status, time_ms, objective, wrong=False):
    return {
        "status": status or "",
        "time_ms": float(time_ms) if time_ms not in ("", None) else MAX_TIME_MS,
        "objective": _parse_obj(objective) if objective else None,
        "wrong": bool(wrong),
    }


EMPTY = make_row("", MAX_TIME_MS, "")


def score_per_instance(entity_inst, opp_inst, instances, kinds):
    """Per-instance Borda score for one entity against the 15 ladder
    opponents. Iterating over all 15 includes self-comparison naturally
    when entity is cp-sat 8c — _compare returns 0.5 on identical solved
    rows and 0 on identical unsolved rows, which is convention #2."""
    out = np.zeros(len(instances))
    for j, key in enumerate(instances):
        kind = kinds.get(key)
        if kind is None:
            continue
        a = entity_inst.get(key, EMPTY)
        a_broken = a["wrong"] or a["status"] == "Error"
        s = 0.0
        for opp in opp_inst:
            b = opp.get(key, EMPTY)
            b_broken = b["wrong"] or b["status"] == "Error"
            if a_broken and b_broken:
                sa = 0.0
            elif a_broken:
                sa = 0.0
            elif b_broken:
                sa = 1.0
            else:
                sa, _ = _compare(
                    a["status"], a["time_ms"], a["objective"],
                    b["status"], b["time_ms"], b["objective"],
                    kind,
                )
            s += sa
        out[j] = s
    return out


def import_bkp():
    spec = importlib.util.spec_from_file_location("bkp", BKP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("k_extra", type=int,
                    help="number of run portfolios on top of the cp-sat(8c) anchor")
    ap.add_argument("--track", choices=("ineligible", "eligible"), default="ineligible")
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    portfolio_csv = ALL_CSV if args.track == "ineligible" else ELIG_CSV
    problem_types = load_problem_types(TYPES_CSV)

    open_rows = list(csv.DictReader(open(OPEN_CSV)))
    instances, inst_year, inst_model = [], {}, {}
    seen = set()
    for r in open_rows:
        k = (r["problem"], r["name"])
        if k not in seen:
            seen.add(k)
            instances.append(k)
            inst_year[k] = r["year"]
            inst_model[k] = r["model"]
    kinds = {k: problem_types.get((k[0], inst_model[k])) for k in instances}

    open_cat = sorted({(r["solver"], int(r["cores"]))
                       for r in open_rows if r["open_category"] == "True"})
    opp_inst = []
    for cfg in open_cat:
        d = {}
        for r in open_rows:
            if (r["solver"], int(r["cores"])) == cfg:
                d[(r["problem"], r["name"])] = make_row(
                    r["status"], r["time_ms"], r["objective"], r["wrong"] == "True")
        opp_inst.append(d)

    cps_inst = {k: v for k, v in
                next(d for cfg, d in zip(open_cat, opp_inst) if cfg == CPSAT_8C).items()}

    portfolio_inst = {}
    for r in csv.DictReader(open(portfolio_csv)):
        name = r["schedule"]
        portfolio_inst.setdefault(name, {})[(r["problem"], r["name"])] = make_row(
            r["status"], r["time_ms"], r["objective"], r["wrong"] == "True")
    schedules = sorted(portfolio_inst)
    print(f"Loaded {len(schedules)} run portfolios from {portfolio_csv.name}")

    fixed_score = score_per_instance(cps_inst, opp_inst, instances, kinds)
    print(f"cp-sat(8c) convention-#2 baseline: {fixed_score.sum():.2f}   (expect 3693.71)")

    portfolios = []
    for name in schedules:
        sv = score_per_instance(portfolio_inst[name], opp_inst, instances, kinds)
        portfolios.append((sv, name))

    bkp = import_bkp()
    portfolios = bkp.prune_portfolios(portfolios, baseline=fixed_score)
    print(f"After dominance pruning: {len(portfolios)} candidates")

    results = bkp.best_k_portfolios(
        portfolios, k=args.k_extra, baseline=fixed_score, top_n=args.top_n)

    print()
    print(f"Track: {args.track}, k = {args.k_extra + 1} "
          f"(cp-sat(8c) + {args.k_extra} run portfolio{'s' if args.k_extra != 1 else ''})")
    print()
    for rank, (s, robustness, names) in enumerate(results, 1):
        print(f"# {rank:2d}  oracle_score={s:.2f}  robustness={robustness:.2f}")
        print(f"    Portfolio 1 (fixed): cp-sat(8c)")
        for i, name in enumerate(names, 2):
            print(f"    Portfolio {i}: {name}")
        print()


if __name__ == "__main__":
    main()
