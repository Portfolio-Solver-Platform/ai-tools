"""
Build training datasets by joining mzn-challenge feature pickles with the
median-rep portfolio benchmarks (cpsat8, k1-8c-8s-v1, ek1-8c-8s-v2).

Reads benchmarks/portfolios/final-portfolios/combined_median.csv, the
aggregate-step output that picks one canonical rep per (portfolio, year,
instance) via the MiniZinc Challenge Borda ordering, with OOM-tainted
instances already removed.

For each DATASET below we compute per-instance Borda scores (pairwise
tournament between the listed portfolios) and save an .npz to data/.
With P portfolios in a dataset each Borda score is in [0, P-1].

Output (one .npz per dataset) with arrays:
  X          (N, F)    feature vectors (float64)
  Y          (N, P)    Borda score per portfolio (training target)
  time_ms    (N, P)    median rep's solve time
  status     (N, P)    median rep's status
  objective  (N, P)    median rep's objective as float (NaN if missing)
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
COMBINED_MEDIAN_CSV = (
    PROJECT_ROOT / "benchmarks/portfolios/final-portfolios/combined_median.csv"
)
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


def load_median_csv(path: Path) -> dict[tuple[str, str], dict[str, dict]]:
    """Load combined_median.csv into {(year_str, feature_key): {portfolio: row}}.
    'row' has the keys borda.pairwise_score needs (status, objective, time_ms)
    plus problem/model/name for metadata."""
    out: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = make_key(r["problem"], r["model"], r["name"])
            out[(r["year"], key)][r["schedule"]] = {
                "problem":   r["problem"],
                "model":     r["model"],
                "name":      r["name"],
                "status":    r["status"],
                "objective": r["objective"],
                "time_ms":   r["time_ms"],
            }
    return out


def load_features(year: int, skipped: dict) -> dict | None:
    feat_path = DATA_DIR / f"mznc{year}_features.pkl"
    if not feat_path.exists():
        skipped["missing_year_features"].append(str(year))
        return None
    with open(feat_path, "rb") as f:
        return pickle.load(f)


def build_dataset(stem: str, portfolios: list[str], problem_types: dict,
                  median_data: dict[tuple[str, str], dict[str, dict]]) -> None:
    print(f"\n=== {stem}  portfolios={portfolios} ===")
    Xs, Ys, times_list, statuses, objectives, metas = [], [], [], [], [], []
    skipped: dict[str, list[str]] = defaultdict(list)
    feature_dim = None

    for year in YEARS:
        features = load_features(year, skipped)
        if features is None:
            continue

        for key, feat_vec in features.items():
            if feat_vec is None:
                skipped["null_features"].append(f"{year}/{key}")
                continue

            per_portfolio = median_data.get((str(year), key), {})
            rows = [per_portfolio.get(p) for p in portfolios]
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
    if not COMBINED_MEDIAN_CSV.exists():
        print(f"missing median CSV: {COMBINED_MEDIAN_CSV}", file=sys.stderr)
        sys.exit(1)
    if not PROBLEM_TYPES_CSV.exists():
        print(f"missing problem types: {PROBLEM_TYPES_CSV}", file=sys.stderr)
        sys.exit(1)

    problem_types = load_problem_types(PROBLEM_TYPES_CSV)
    median_data = load_median_csv(COMBINED_MEDIAN_CSV)
    print(f"Loaded {len(median_data)} (year, instance) groups from "
          f"{COMBINED_MEDIAN_CSV.name}")
    for stem, portfolios in DATASETS:
        build_dataset(stem, portfolios, problem_types, median_data)


if __name__ == "__main__":
    main()
