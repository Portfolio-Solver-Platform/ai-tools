#!/usr/bin/env python3
"""Generate a Typst/Lilaq plot showing time saved by bound-sharing interval vs. no sharing (best-static portfolio)."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "parasol-benchmarks" / "best-static-bound-sharing"
OUTPUT_FILE = Path(__file__).resolve().parent / "best_static_total_time.typ"

BASELINE = ("none", "best-static-bound-sharing-2025-none")
INTERVALS = [
    (2,  "best-static-bound-sharing-2025-2s"),
    (4,  "best-static-bound-sharing-2025-4s"),
    (8,  "best-static-bound-sharing-2025-8s"),
    (16, "best-static-bound-sharing-2025-16s"),
    (32, "best-static-bound-sharing-2025-32s"),
    (64, "best-static-bound-sharing-2025-64s"),
]


def read_total_time_s(csv_path: Path) -> float:
    total_ms = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            total_ms += int(row["time_ms"])
    return total_ms / 1000.0


def main():
    baseline_path = DATA_DIR / BASELINE[1] / "results.csv"
    baseline_s = read_total_time_s(baseline_path)
    print(f" none (baseline): {baseline_s:,.1f} s")

    xs = []
    deltas = []
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

    xs_typ = "(" + ", ".join(str(x) for x in xs) + ")"
    deltas_typ = "(" + ", ".join(f"{d:.1f}" for d in deltas) + ")"
    ticks_typ = "(" + ", ".join(f'({x}, [{x}s])' for x in xs) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = {xs_typ}
#let deltas = {deltas_typ}

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Time Saved vs.\\ No Bound Sharing (best-static, baseline: {baseline_s:,.0f}s)],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xscale: "log",
  xaxis: (
    ticks: {ticks_typ},
    subticks: none,
  ),

  // Reference line at 0
  lq.plot((2, 64), (0, 0), stroke: (dash: "dashed", paint: black)),

  // Data
  lq.plot(
    xs, deltas,
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
