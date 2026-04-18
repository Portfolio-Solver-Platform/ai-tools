#!/usr/bin/env python3
"""
Borda scores for k1-8c-8s-v1 bound-sharing configs vs. all solvers from combined.csv (2025).

All solvers participate in pairwise scoring, but only bound-sharing configs are plotted.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parent.parent.parent / "scoring"
sys.path.insert(0, str(SCORING_DIR))

from borda import load_problem_types, pairwise_score

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "open-category-benchmarks" / "combined.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "k1-8c-8s-v1"
TYPES_CSV = Path(__file__).resolve().parent.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "combined_borda_2025.typ"
YEAR = "2025"

CONFIGS = [
    ("none", "k1-8c-8s-v1-bound-sharing-2025-none"),
    ("2s",   "k1-8c-8s-v1-bound-sharing-2025-2s"),
    ("4s",   "k1-8c-8s-v1-bound-sharing-2025-4s"),
    ("8s",   "k1-8c-8s-v1-bound-sharing-2025-8s"),
    ("16s",  "k1-8c-8s-v1-bound-sharing-2025-16s"),
    ("32s",  "k1-8c-8s-v1-bound-sharing-2025-32s"),
    ("64s",  "k1-8c-8s-v1-bound-sharing-2025-64s"),
    ("96s",  "k1-8c-8s-v1-bound-sharing-2025-96s"),
    ("128s", "k1-8c-8s-v1-bound-sharing-2025-128s"),
    ("160s", "k1-8c-8s-v1-bound-sharing-2025-160s"),
    ("192s", "k1-8c-8s-v1-bound-sharing-2025-192s"),
    ("224s", "k1-8c-8s-v1-bound-sharing-2025-224s"),
    ("256s", "k1-8c-8s-v1-bound-sharing-2025-256s"),
]


def load_bound_sharing_rows(label: str, folder: Path) -> list[dict]:
    rows = []
    with open(folder / "results.csv") as f:
        for row in csv.DictReader(f):
            rows.append({
                "solver": f"bs-{label}",
                "problem": row["problem"],
                "name": row["name"],
                "model": row["model"],
                "time_ms": row["time_ms"],
                "objective": row["objective"],
                "status": row["optimal"],
            })
    return rows


def main():
    problem_types = load_problem_types(TYPES_CSV)

    all_rows = []
    with open(CSV_PATH) as f:
        all_rows = [r for r in csv.DictReader(f) if r["year"] == YEAR]

    for label, folder_name in CONFIGS:
        folder = DATA_DIR / folder_name
        if not (folder / "results.csv").exists():
            print(f"Warning: {folder}/results.csv not found, skipping")
            continue
        all_rows.extend(load_bound_sharing_rows(label, folder))

    instances: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["problem"], r["model"], r["name"])
        instances[key].append(r)

    scores: dict[str, float] = defaultdict(float)
    unknown_types: set[str] = set()

    for (problem, model, name), group in instances.items():
        kind = problem_types.get((problem, model))
        if kind is None:
            unknown_types.add(f"{problem}/{model}")
            continue
        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                scores[s["solver"]] += pairwise_score(s, s2, kind)

    if unknown_types:
        print(f"WARNING: {len(unknown_types)} models not found in problem_types.csv:")
        for m in sorted(unknown_types):
            print(f"  {m}")
        print()

    print("All solver Borda scores (2025):")
    for solver, score in sorted(scores.items(), key=lambda x: -x[1]):
        marker = " <--" if solver.startswith("bs-") else ""
        print(f"  {solver:<30s} {score:>10.2f}{marker}")

    labels = [label for label, _ in CONFIGS]
    bs_scores = [scores.get(f"bs-{label}", 0) for label in labels]

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    values_typ = "(" + ", ".join(f"{v:.2f}" for v in bs_scores) + ")"

    spread = max(bs_scores) - min(bs_scores)
    pad = max(spread * 0.20, 1.0)
    ylim_low = max(0, min(bs_scores) - pad)
    ylim_high = max(bs_scores) + pad

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let scores = {values_typ}

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score: k1-8c-8s-v1 Bound-Sharing (vs.\\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: ({ylim_low:.1f}, {ylim_high:.1f}),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
"""

    OUTPUT_FILE.write_text(typst)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
