#!/usr/bin/env python3
"""Generate a Typst/Lilaq plot comparing Borda scores across bound-sharing intervals (cpsat-chuffed)."""

import csv
import sys
from collections import defaultdict
from pathlib import Path

SCORING_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scoring"
sys.path.insert(0, str(SCORING_DIR))

from borda import load_problem_types, pairwise_score

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "parasol-benchmarks" / "new-bound-sharing" / "cpsat-chuffed-bound-sharing"
TYPES_CSV = Path(__file__).resolve().parent.parent.parent.parent / "open-category-benchmarks" / "problem_types.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "bound_sharing_borda.typ"

CONFIGS = [
    ("none", "cpsat-chuffed-bound-sharing-2025-none"),
    ("2s",   "cpsat-chuffed-bound-sharing-2025-2s"),
    ("4s",   "cpsat-chuffed-bound-sharing-2025-4s"),
    ("8s",   "cpsat-chuffed-bound-sharing-2025-8s"),
    ("16s",  "cpsat-chuffed-bound-sharing-2025-16s"),
    ("32s",  "cpsat-chuffed-bound-sharing-2025-32s"),
    ("64s",  "cpsat-chuffed-bound-sharing-2025-64s"),
    ("300s", "cpsat-chuffed-bound-sharing-2025-300s"),
]


def load_config_rows(folder: Path, config_label: str) -> list[dict]:
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
                "status": row["optimal"],
            })
    return rows


def main():
    problem_types = load_problem_types(TYPES_CSV)

    all_rows = []
    for label, folder_name in CONFIGS:
        folder = DATA_DIR / folder_name
        if not (folder / "results.csv").exists():
            print(f"Warning: {folder}/results.csv not found, skipping")
            continue
        all_rows.extend(load_config_rows(folder, label))

    instances: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        key = (r["problem"], r["name"], r["model"])
        instances[key].append(r)

    scores: dict[str, float] = defaultdict(float)
    unknown_types: set[str] = set()

    for inst_key, group in instances.items():
        problem = group[0]["problem"]
        model = group[0]["model"]
        kind = problem_types.get((problem, model))
        if kind is None:
            unknown_types.add(f"{problem}/{model}")
            continue
        for i, s in enumerate(group):
            for j, s2 in enumerate(group):
                if i == j:
                    continue
                scores[s["config"]] += pairwise_score(s, s2, kind)

    if unknown_types:
        print(f"WARNING: {len(unknown_types)} models not found in problem_types.csv:")
        for m in sorted(unknown_types):
            print(f"  {m}")
        print()

    print("Borda scores by bound-sharing interval (cpsat-chuffed):")
    for label, _ in CONFIGS:
        print(f"  {label:>5s}: {scores.get(label, 0):,.2f}")

    labels = [label for label, _ in CONFIGS]
    values = [scores.get(label, 0) for label in labels]

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    values_typ = "(" + ", ".join(f"{v:.2f}" for v in values) + ")"

    spread = max(values) - min(values)
    pad = max(spread * 0.20, 1.0)
    ylim_low = max(0, min(values) - pad)
    ylim_high = max(values) + pad

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let scores = {values_typ}

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (cp-sat + chuffed)],
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
