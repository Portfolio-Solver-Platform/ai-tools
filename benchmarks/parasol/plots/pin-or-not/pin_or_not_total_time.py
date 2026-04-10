#!/usr/bin/env python3
"""Generate a Typst/Lilaq grouped bar chart comparing total solving time: pin vs not-pin for each portfolio."""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "parasol-benchmarks" / "pin-or-not"
OUTPUT_FILE = Path(__file__).resolve().parent / "pin_or_not_total_time.typ"

PORTFOLIOS = [
    ("choco-cpsat", "choco-cpsat-pin", "choco-cpsat-not-pin"),
    ("choco-highs", "choco-highs-pin", "choco-highs-not-pin"),
]


def read_total_time_s(csv_path: Path) -> float:
    total_ms = 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            total_ms += int(row["time_ms"])
    return total_ms / 1000.0


def main():
    labels = []
    pin_times = []
    not_pin_times = []

    for label, pin_folder, not_pin_folder in PORTFOLIOS:
        pin_csv = DATA_DIR / pin_folder / "results.csv"
        not_pin_csv = DATA_DIR / not_pin_folder / "results.csv"
        if not pin_csv.exists() or not not_pin_csv.exists():
            print(f"Warning: missing data for {label}, skipping")
            continue
        t_pin = read_total_time_s(pin_csv)
        t_not = read_total_time_s(not_pin_csv)
        labels.append(label)
        pin_times.append(t_pin)
        not_pin_times.append(t_not)
        print(f"{label:<15s} pin={t_pin:>10,.1f}s   not-pin={t_not:>10,.1f}s   diff={t_pin-t_not:>+8,.1f}s")

    labels_typ = "(" + ", ".join(f'"{l}"' for l in labels) + ")"
    pin_typ = "(" + ", ".join(f"{t:.1f}" for t in pin_times) + ")"
    not_pin_typ = "(" + ", ".join(f"{t:.1f}" for t in not_pin_times) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = {labels_typ}
#let pin-times = {pin_typ}
#let not-pin-times = {not_pin_typ}

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Total Solving Time: Pin vs.\\ No-Pin],
  ylabel: [Total time (s)],
  xlabel: [Portfolio],
  legend: (position: (100% + .5em, 0%)),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), pin-times,
    offset: -0.2, width: 0.4,
    fill: blue.lighten(30%),
    label: [pin],
  ),
  lq.bar(
    range(labels.len()), not-pin-times,
    offset: 0.2, width: 0.4,
    fill: orange.lighten(30%),
    label: [not-pin],
  ),
)
"""

    OUTPUT_FILE.write_text(typst)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
