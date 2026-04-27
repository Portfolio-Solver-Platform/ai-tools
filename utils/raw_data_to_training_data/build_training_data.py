"""
Build training datasets by joining mzn-challenge feature pickles with the
portfolio benchmarks (cpsat8, k1-8c-8s-v1, ek1-8c-8s-v2).

For each DATASET below we compute per-instance Borda scores (MiniZinc-
challenge style: pairwise tournament between the listed portfolios) and save
an .npz to data/. With P portfolios in a dataset each Borda score is in [0, P-1].

Output (one .npz per dataset) with arrays:
  X          (N, F)    feature vectors (float64)
  Y          (N, P)    Borda score per portfolio (training target)
  time_ms    (N, P)    raw solve time
  status     (N, P)    "Optimal" | "Unsat" | "Unknown"
  objective  (N, P)    objective value as float (NaN if missing)
  meta       (N,)      structured: year:int, problem:str, model:str, name:str
  portfolios (P,)      portfolio names matching column order

Rows are dropped (with counts reported) when any of these is missing:
- features for the (year, instance)
- a result row in any portfolio of the dataset
- a kind (SAT/MIN/MAX) for (problem, model) in problem_types.csv
"""
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "benchmarks/portfolios/final-portfolios/portfolios-final"
PROBLEM_TYPES_CSV = PROJECT_ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
SCORING_DIR = PROJECT_ROOT / "benchmarks/scoring"
sys.path.insert(0, str(SCORING_DIR))
from borda import load_problem_types, pairwise_score  # noqa: E402

YEARS = range(2011, 2026)

# (output stem, portfolio list)  →  data/{stem}_training_data.npz
DATASETS: list[tuple[str, list[str]]] = [
    ("portfolios_cpsat8_k1",      ["cpsat8", "k1-8c-8s-v1"]),
    ("portfolios_cpsat8_ek1",     ["cpsat8", "ek1-8c-8s-v2"]),
    ("portfolios_cpsat8_k1_ek1",  ["cpsat8", "k1-8c-8s-v1", "ek1-8c-8s-v2"]),
]


def make_key(problem: str, model: str, name: str) -> str:
    return f"{problem}_{model}_" if model == name else f"{problem}_{model}_{name}"


def load_results(csv_path: Path) -> dict[str, dict]:
    """Load results.csv keyed by feature key. Adds 'status' alias for 'optimal'
    so rows are usable with borda.pairwise_score directly."""
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["status"] = r["optimal"]
            out[make_key(r["problem"], r["model"], r["name"])] = r
    return out


def load_year_data(year: int, portfolios: list[str], skipped: dict) -> tuple[dict, dict] | None:
    feat_path = DATA_DIR / f"mznc{year}_features.pkl"
    if not feat_path.exists():
        skipped["missing_year_features"].append(str(year))
        return None
    with open(feat_path, "rb") as f:
        features = pickle.load(f)

    portfolio_results = {}
    for p in portfolios:
        csv_path = RESULTS_ROOT / p / f"{p}-{year}" / "results.csv"
        if not csv_path.exists():
            skipped["missing_year_results"].append(f"{year}/{p}")
            return None
        portfolio_results[p] = load_results(csv_path)
    return features, portfolio_results


def build_dataset(stem: str, portfolios: list[str], problem_types: dict) -> None:
    print(f"\n=== {stem}  portfolios={portfolios} ===")
    Xs, Ys, times_list, statuses, objectives, metas = [], [], [], [], [], []
    skipped: dict[str, list[str]] = defaultdict(list)
    feature_dim = None

    for year in YEARS:
        loaded = load_year_data(year, portfolios, skipped)
        if loaded is None:
            continue
        features, portfolio_results = loaded

        for key, feat_vec in features.items():
            if feat_vec is None:
                skipped["null_features"].append(f"{year}/{key}")
                continue

            rows = [portfolio_results[p].get(key) for p in portfolios]
            missing = [p for p, r in zip(portfolios, rows) if r is None]
            if missing:
                skipped["missing_in_portfolio"].append(f"{year}/{key} (missing in {','.join(missing)})")
                continue

            ref = rows[0]
            problem, model, name = ref["problem"], ref["model"], ref["name"]
            kind = problem_types.get((problem, model))
            if kind is None:
                skipped["unknown_problem_type"].append(f"{year}/{problem}/{model}")
                continue

            x = np.asarray(feat_vec, dtype=np.float64).reshape(-1)
            if feature_dim is None:
                feature_dim = x.shape[0]
            elif x.shape[0] != feature_dim:
                skipped["bad_feature_shape"].append(f"{year}/{key} (got {x.shape[0]}, expected {feature_dim})")
                continue

            borda = np.zeros(len(portfolios), dtype=np.float64)
            for i, s in enumerate(rows):
                for j, s2 in enumerate(rows):
                    if i == j:
                        continue
                    borda[i] += pairwise_score(s, s2, kind)

            times = np.array([float(r["time_ms"]) for r in rows], dtype=np.float64)
            stats = np.array([r["status"] for r in rows], dtype=object)
            objs = np.array(
                [float(r["objective"]) if r["objective"] != "" else np.nan for r in rows],
                dtype=np.float64,
            )

            Xs.append(x)
            Ys.append(borda)
            times_list.append(times)
            statuses.append(stats)
            objectives.append(objs)
            metas.append((year, problem, model, name))

    if not Xs:
        print(f"  no rows produced for {stem} — skipping save")
        return

    X = np.stack(Xs)
    Y = np.stack(Ys)
    time_ms = np.stack(times_list)
    status = np.stack(statuses)
    objective = np.stack(objectives)

    max_problem = max(len(m[1]) for m in metas)
    max_model = max(len(m[2]) for m in metas)
    max_name = max(len(m[3]) for m in metas)
    meta_dtype = np.dtype([
        ("year", np.int32),
        ("problem", f"U{max_problem}"),
        ("model", f"U{max_model}"),
        ("name", f"U{max_name}"),
    ])
    meta = np.array(metas, dtype=meta_dtype)

    out_path = DATA_DIR / f"{stem}_training_data.npz"
    np.savez(
        out_path,
        X=X, Y=Y, time_ms=time_ms, status=status, objective=objective, meta=meta,
        portfolios=np.array(portfolios),
    )

    print(f"  wrote {out_path}")
    print(f"  shapes: X={X.shape}  Y={Y.shape}  time_ms={time_ms.shape}")
    print(f"  Y per portfolio (mean / max): {list(zip(portfolios, Y.mean(axis=0).round(3), Y.max(axis=0)))}")
    if skipped:
        total = sum(len(v) for v in skipped.values())
        print(f"  skipped: {total} total")
        for reason, items in sorted(skipped.items()):
            print(f"    {reason}: {len(items)}")


def main():
    if not RESULTS_ROOT.is_dir():
        print(f"missing results: {RESULTS_ROOT}", file=sys.stderr)
        sys.exit(1)
    if not PROBLEM_TYPES_CSV.exists():
        print(f"missing problem types: {PROBLEM_TYPES_CSV}", file=sys.stderr)
        sys.exit(1)

    problem_types = load_problem_types(PROBLEM_TYPES_CSV)
    for stem, portfolios in DATASETS:
        build_dataset(stem, portfolios, problem_types)


if __name__ == "__main__":
    main()
