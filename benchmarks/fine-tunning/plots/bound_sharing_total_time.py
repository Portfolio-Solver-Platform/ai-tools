#!/usr/bin/env python3
"""Generate a Typst/Lilaq plot showing time saved by bound-sharing interval vs. no sharing (k1-8c-8s-v1)."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "k1-8c-8s-v1"
OUTPUT_FILE = Path(__file__).resolve().parent / "bound_sharing_total_time.typ"

BASELINE = ("none", "k1-8c-8s-v1-bound-sharing-2025-none")
INTERVALS = [
    (2,   "k1-8c-8s-v1-bound-sharing-2025-2s"),
    (4,   "k1-8c-8s-v1-bound-sharing-2025-4s"),
    (8,   "k1-8c-8s-v1-bound-sharing-2025-8s"),
    (16,  "k1-8c-8s-v1-bound-sharing-2025-16s"),
    (32,  "k1-8c-8s-v1-bound-sharing-2025-32s"),
    (64,  "k1-8c-8s-v1-bound-sharing-2025-64s"),
    (96,  "k1-8c-8s-v1-bound-sharing-2025-96s"),
    (128, "k1-8c-8s-v1-bound-sharing-2025-128s"),
    (160, "k1-8c-8s-v1-bound-sharing-2025-160s"),
    (192, "k1-8c-8s-v1-bound-sharing-2025-192s"),
    (224, "k1-8c-8s-v1-bound-sharing-2025-224s"),
    (256, "k1-8c-8s-v1-bound-sharing-2025-256s"),
]


def read_total_time_s(csv_path: Path) -> float:
    total_ms = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            total_ms += int(row["time_ms"])
    return total_ms / 1000.0


def main():
    baseline_s = read_total_time_s(DATA_DIR / BASELINE[1] / "results.csv")
    print(f" none (baseline): {baseline_s:,.1f} s")

    xs, deltas = [], []
    for interval, folder in INTERVALS:
        csv_path = DATA_DIR / folder / "results.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        total_s = read_total_time_s(csv_path)
        delta = baseline_s - total_s
        xs.append(interval)
        deltas.append(delta)
        print(f"  {interval:>3d}s: {total_s:,.1f} s  (saved {delta:,.1f} s)")

    labels = [f"{x}s" for x in xs]

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    deltas_typ = "(" + ", ".join(f"{d:.1f}" for d in deltas) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let deltas = {deltas_typ}

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Time Saved vs.\\ No Bound Sharing (k1-8c-8s-v1, baseline: {baseline_s:,.0f}s)],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.plot((0, {len(xs) - 1}), (0, 0), stroke: (dash: "dashed", paint: black)),

  lq.plot(
    range(labels.len()), deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
"""

    OUTPUT_FILE.write_text(typst)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
