"""
Like build_training_data.py, but enriches each instance with multi-rep
variance features and a per-instance "label confidence" score.

The existing pipeline takes the median rep per (portfolio, year, instance)
and discards the other two reps. Here we ALSO compute, from the full
per-rep data in combined.csv:

  per-portfolio rep-variance features (4 numeric per portfolio):
    log_iqr           = log1p(q75 - q25 of time_ms)
    log_range         = log1p(max - min of time_ms)
    log_std           = log1p(std of time_ms)
    agree_optimal     = fraction of 3 reps reporting 'Optimal'

  per-instance label_confidence (binary tournament):
    For every cross-rep pairing (rep_i of A vs rep_j of B) we apply the
    same pairwise score the median selection uses. Fraction of the 9
    pairings that agree with the median's binary winner = label_confidence.
    1.0 = unambiguous, 0.5 = coin-flip, 0 = always contradicted.

Output: data/{stem}_training_data_reps.npz with the original arrays plus
  X_rep_features    (N, P, 4)
  label_confidence  (N,)
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pickle as _serialiser

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMBINED_FULL_CSV = (
    PROJECT_ROOT / "benchmarks/portfolios/final-portfolios/combined.csv"
)
COMBINED_MEDIAN_CSV = (
    PROJECT_ROOT / "benchmarks/portfolios/final-portfolios/combined_median.csv"
)
PROBLEM_TYPES_CSV = PROJECT_ROOT / "benchmarks/open-category-benchmarks/problem_types.csv"
SCORING_DIR = PROJECT_ROOT / "benchmarks/scoring"
sys.path.insert(0, str(SCORING_DIR))
from borda import load_problem_types, pairwise_score  # noqa: E402

YEARS = range(2011, 2026)

DATASETS: list[tuple[str, list[str]]] = [
    ("portfolios_cpsat8_k1",  ["cpsat8", "k1-8c-8s-v1"]),
    ("portfolios_cpsat8_ek1", ["cpsat8", "ek1-8c-8s-v2"]),
]


def make_key(problem: str, model: str, name: str) -> str:
    return f"{problem}_{model}_" if model == name else f"{problem}_{model}_{name}"


def load_full_rep_csv(path: Path):
    out: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with open(path) as f:
        for r in csv.DictReader(f):
            key = make_key(r["problem"], r["model"], r["name"])
            out[(r["year"], key)][r["schedule"]].append({
                "rep":       r["rep"],
                "status":    r["status"],
                "objective": r["objective"],
                "time_ms":   r["time_ms"],
                "problem":   r["problem"],
                "model":     r["model"],
                "name":      r["name"],
            })
    return out


def load_median_csv(path: Path):
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


def rep_features(rep_rows: list[dict]) -> np.ndarray:
    times = np.array([float(r["time_ms"]) for r in rep_rows], dtype=np.float64)
    log_iqr   = float(np.log1p(np.percentile(times, 75) - np.percentile(times, 25)))
    log_range = float(np.log1p(times.max() - times.min()))
    log_std   = float(np.log1p(times.std()))
    agree_opt = float(np.mean([r["status"] == "Optimal" for r in rep_rows]))
    return np.array([log_iqr, log_range, log_std, agree_opt])


def label_confidence(rep_per_p: dict[str, list[dict]], portfolios: list[str],
                    median_borda: np.ndarray, kind: str) -> float:
    if len(portfolios) != 2:
        return 1.0
    if median_borda[0] == median_borda[1]:
        return 0.5
    a_winner = 1 if median_borda[0] > median_borda[1] else 0
    p1, p2 = portfolios
    rows1 = rep_per_p.get(p1, [])
    rows2 = rep_per_p.get(p2, [])
    if not rows1 or not rows2:
        return 0.0
    agree = 0.0; total = 0
    for r1 in rows1:
        for r2 in rows2:
            s12 = pairwise_score(r1, r2, kind)
            if s12 > 0.5:
                this = 1
            elif s12 < 0.5:
                this = 0
            else:
                this = -1
            if this == a_winner:
                agree += 1
            elif this == -1:
                agree += 0.5
            total += 1
    return float(agree) / total if total else 0.0


def build_dataset(stem: str, portfolios: list[str], problem_types,
                  median_data, full_rep_data):
    print(f"\n=== {stem}  portfolios={portfolios} ===")
    Xs, Ys, Xreps, confs = [], [], [], []
    times_list, statuses, objectives, metas = [], [], [], []
    feature_dim = None

    for year in YEARS:
        feat_path = DATA_DIR / f"mznc{year}_features.pkl"
        if not feat_path.exists():
            continue
        with open(feat_path, "rb") as f:
            features = _serialiser.load(f)

        for key, feat_vec in features.items():
            if feat_vec is None:
                continue
            per_p = median_data.get((str(year), key), {})
            rep_per_p = full_rep_data.get((str(year), key), {})
            rows = [per_p.get(p) for p in portfolios]
            if any(r is None for r in rows):
                continue

            ref = rows[0]
            problem, model, name = ref["problem"], ref["model"], ref["name"]
            kind = problem_types.get((problem, model))
            if kind is None:
                continue

            x = np.asarray(feat_vec, dtype=np.float64).reshape(-1)
            if feature_dim is None:
                feature_dim = x.shape[0]
            elif x.shape[0] != feature_dim:
                continue

            borda = np.zeros(len(portfolios), dtype=np.float64)
            for i, s in enumerate(rows):
                for j, s2 in enumerate(rows):
                    if i == j:
                        continue
                    borda[i] += pairwise_score(s, s2, kind)

            xrep = np.stack([rep_features(rep_per_p.get(p, [])) for p in portfolios])
            conf = label_confidence(rep_per_p, portfolios, borda, kind)

            times = np.array([float(r["time_ms"]) for r in rows], dtype=np.float64)
            stats = np.array([r["status"] for r in rows], dtype=object)
            objs = np.array(
                [float(r["objective"]) if r["objective"] != "" else np.nan for r in rows],
                dtype=np.float64,
            )

            Xs.append(x); Ys.append(borda); Xreps.append(xrep); confs.append(conf)
            times_list.append(times); statuses.append(stats); objectives.append(objs)
            metas.append((year, problem, model, name))

    if not Xs:
        print("  no rows; skipping")
        return

    X = np.stack(Xs); Y = np.stack(Ys)
    X_rep_features = np.stack(Xreps)
    label_conf = np.array(confs)
    time_ms = np.stack(times_list); status = np.stack(statuses); objective = np.stack(objectives)
    max_problem = max(len(m[1]) for m in metas)
    max_model = max(len(m[2]) for m in metas)
    max_name = max(len(m[3]) for m in metas)
    meta_dtype = np.dtype([
        ("year", np.int32), ("problem", f"U{max_problem}"),
        ("model", f"U{max_model}"), ("name", f"U{max_name}"),
    ])
    meta = np.array(metas, dtype=meta_dtype)

    out_path = DATA_DIR / f"{stem}_training_data_reps.npz"
    np.savez(
        out_path, X=X, Y=Y,
        X_rep_features=X_rep_features, label_confidence=label_conf,
        time_ms=time_ms, status=status, objective=objective, meta=meta,
        portfolios=np.array(portfolios),
    )
    print(f"  wrote {out_path}")
    print(f"  X={X.shape}  Y={Y.shape}  X_rep_features={X_rep_features.shape}")
    print(f"  label_confidence: mean={label_conf.mean():.3f} min={label_conf.min():.3f} max={label_conf.max():.3f}")
    print(f"  confidence quantiles q10/q25/q50/q75/q90: " +
          " / ".join(f"{np.percentile(label_conf, q):.3f}" for q in [10,25,50,75,90]))
    print(f"  rep-feature stats per portfolio:")
    for i, p in enumerate(portfolios):
        feats = X_rep_features[:, i, :]
        print(f"    {p}: log_iqr mean={feats[:,0].mean():.2f}  "
              f"agree_optimal mean={feats[:,3].mean():.3f}")


def main():
    if not COMBINED_FULL_CSV.exists():
        print(f"missing: {COMBINED_FULL_CSV}"); sys.exit(1)
    if not COMBINED_MEDIAN_CSV.exists():
        print(f"missing: {COMBINED_MEDIAN_CSV}"); sys.exit(1)

    problem_types = load_problem_types(PROBLEM_TYPES_CSV)
    median_data = load_median_csv(COMBINED_MEDIAN_CSV)
    full_rep_data = load_full_rep_csv(COMBINED_FULL_CSV)
    print(f"Loaded {len(median_data)} groups from median CSV")
    print(f"Loaded {len(full_rep_data)} groups from full CSV")
    for stem, portfolios in DATASETS:
        build_dataset(stem, portfolios, problem_types, median_data, full_rep_data)


if __name__ == "__main__":
    main()
