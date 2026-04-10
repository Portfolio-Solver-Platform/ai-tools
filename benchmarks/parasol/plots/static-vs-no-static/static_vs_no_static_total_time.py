#!/usr/bin/env python3
"""Generate a Typst/Lilaq bar chart comparing total solving time: static vs no-static."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "parasol-benchmarks" / "static-vs-no-static"
OUTPUT_FILE = Path(__file__).resolve().parent / "static_vs_no_static_total_time.typ"

CONFIGS = [
    ("static",    "static"),
    ("no-static", "no-static"),
]


def read_total_time_s(csv_path: Path) -> float:
    total_ms = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            total_ms += int(row["time_ms"])
    return total_ms / 1000.0


def main():
    labels = []
    times = []

    for label, folder in CONFIGS:
        csv_path = DATA_DIR / folder / "results.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping")
            continue
        total_s = read_total_time_s(csv_path)
        labels.append(label)
        times.append(total_s)
        print(f"{label:>10s}: {total_s:,.1f} s")

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    times_typ = "(" + ", ".join(f"{t:.1f}" for t in times) + ")"

    # Zoomed y-axis: pad the min/max by ~2x their spread so the difference is visible.
    lo, hi = min(times), max(times)
    spread = hi - lo
    ymin = lo - spread * 2
    ymax = hi + spread * 2

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let times = {times_typ}

#lq.diagram(
  width: 8cm,
  height: 6cm,
  title: [Total Solving Time: Static vs.\\ No-Static],
  ylabel: [Total time (s)],
  ylim: ({ymin:.1f}, {ymax:.1f}),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), times,
    fill: blue.lighten(30%),
  ),
)
"""

    OUTPUT_FILE.write_text(typst)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
