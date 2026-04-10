#!/usr/bin/env python3
"""Generate a Typst/Lilaq bar chart comparing total solving time: pin vs no-pin."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "pin-degrades-performance"
OUTPUT_FILE = Path(__file__).resolve().parent / "pin_total_time.typ"

CONFIGS = [
    ("pin",    "pin"),
    ("no-pin", "no-pin"),
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
        print(f"{label:>6s}: {total_s:,.1f} s")

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    times_typ = "(" + ", ".join(f"{t:.1f}" for t in times) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let times = {times_typ}

#lq.diagram(
  width: 8cm,
  height: 6cm,
  title: [Total Solving Time: Pin vs.\\ No-Pin],
  ylabel: [Total time (s)],
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
