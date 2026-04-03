#!/usr/bin/env python3
"""Generate a Typst/Lilaq plot comparing Borda scores across bound-sharing intervals."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

# Add scoring dir to path so we can import the shared module
SCORING_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scoring"
sys.path.insert(0, str(SCORING_DIR))

from borda import load_problem_types, pairwise_score

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "cpsat-yuck-bound-sharing"
TYPES_CSV = SCORING_DIR / "problem_types.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "bound_sharing_borda.typ"

CONFIGS = [
    ("none", "cpsat-yuck-bound-sharing-2025-none"),
    ("2s",   "cpsat-yuck-bound-sharing-2025-2s"),
    ("4s",   "cpsat-yuck-bound-sharing-2025-4s"),
    ("8s",   "cpsat-yuck-bound-sharing-2025-8s"),
    ("16s",  "cpsat-yuck-bound-sharing-2025-16s"),
    ("32s",  "cpsat-yuck-bound-sharing-2025-32s"),
]


def load_config_rows(folder: Path, config_label: str) -> list[dict]:
    """Load results.csv and normalize columns to match borda.py expectations."""
    rows = []
    with open(folder / "results.csv") as f:
        for row in csv.DictReader(f):
            rows.append({
                "config": config_label,
                "problem": row["problem"],
                "name": row["name"],
                "model": row["model"],
                "time_ms": row["time_ms"],
                "objective": row["objective"],
                "status": row["optimal"],  # rename optimal -> status
            })
    return rows


def main():
    problem_types = load_problem_types(TYPES_CSV)

    # Load all configs into one list
    all_rows = []
    for label, folder_name in CONFIGS:
        folder = DATA_DIR / folder_name
        if not (folder / "results.csv").exists():
            print(f"Warning: {folder}/results.csv not found, skipping")
            continue
        all_rows.extend(load_config_rows(folder, label))

    # Group by instance (problem, name, model)
    instances: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["problem"], r["name"], r["model"])
        instances[key].append(r)

    # Compute Borda scores per config
    scores: dict[str, float] = defaultdict(float)
    unknown_types: set[str] = set()

    for inst_key, group in instances.items():
        model = group[0]["model"]
        kind = problem_types.get(model)
        if kind is None:
            unknown_types.add(model)
            continue

        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                score = pairwise_score(s, s2, kind)
                scores[s["config"]] += score

    if unknown_types:
        print(f"WARNING: {len(unknown_types)} models not found in problem_types.csv:")
        for m in sorted(unknown_types):
            print(f"  {m}")
        print()

    # Print scores
    print("Borda scores by bound-sharing interval:")
    for label, _ in CONFIGS:
        print(f"  {label:>5s}: {scores.get(label, 0):,.2f}")

    # Generate Typst plot — bar chart with categorical labels
    labels = [label for label, _ in CONFIGS]
    values = [scores.get(label, 0) for label in labels]

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    values_typ = "(" + ", ".join(f"{v:.2f}" for v in values) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let scores = {values_typ}

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
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
