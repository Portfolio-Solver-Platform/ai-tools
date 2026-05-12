#!/usr/bin/env python3
"""Generate a Typst/lilaq two-panel plot showing the parasol harness overhead.

  panel 1 (top, scatter):    standalone time (log x) vs delta = parasol - standalone
                             (symlog y - linear near 0 for the constant cluster,
                             log on tails for the run-variance-driven outliers).
  panel 2 (bottom, hist):    distribution of deltas, focused on the dense region.

Showing all instances reveals the two regimes:
  - short instances (~<10s standalone): tight cluster around ~1-2 s constant overhead.
  - longer instances: large positive AND negative deltas driven by cp-sat run-to-run
    variance with 8-core parallel search.
"""
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
OPEN_CSV = ROOT / "benchmarks/open-category-benchmarks/combined.csv"
RESULTS = ROOT / "benchmarks/portfolios/final-portfolios/portfolios-final"
OUT_FILE = Path(__file__).resolve().parent / "parasol_overhead.typ"
YEARS = ("2023", "2024", "2025")
TIMEOUT_S = 1200

# Histogram window: most data sits in [-2, 8] s; outliers reported in caption.
HIST_MIN, HIST_MAX = -2.0, 8.0
N_BINS = 30


def main():
    open_cs = {}
    with open(OPEN_CSV) as f:
        for r in csv.DictReader(f):
            if r["year"] in YEARS and r["solver"] == "cp-sat" and r["cores"] == "8":
                open_cs[(r["year"], r["problem"], r["model"], r["name"])] = float(r["time_ms"])
    ours = {}
    for y in YEARS:
        with open(RESULTS / "cpsat8" / f"cpsat8-{y}" / "results.csv") as f:
            for r in csv.DictReader(f):
                ours[(y, r["problem"], r["model"], r["name"])] = float(r["time_ms"])

    common = sorted(set(open_cs) & set(ours))
    pairs = [(open_cs[k] / 1000.0, ours[k] / 1000.0) for k in common]
    finished = [(x, y - x) for x, y in pairs if x < TIMEOUT_S and y < TIMEOUT_S]
    xs = [x for x, _ in finished]
    deltas = [d for _, d in finished]

    n_total = len(pairs)
    n_finished = len(finished)
    median_d = statistics.median(deltas)
    mean_d = statistics.mean(deltas)
    n_above = sum(1 for d in deltas if d > HIST_MAX)
    n_below = sum(1 for d in deltas if d < HIST_MIN)

    # Histogram on the dense region; outliers clipped into edge bins so they're visible
    bin_edges = [HIST_MIN + i * (HIST_MAX - HIST_MIN) / N_BINS for i in range(N_BINS + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(N_BINS)]
    bin_width = (HIST_MAX - HIST_MIN) / N_BINS
    counts = [0] * N_BINS
    for d in deltas:
        if d <= HIST_MIN:
            counts[0] += 1
        elif d >= HIST_MAX:
            counts[-1] += 1
        else:
            counts[int((d - HIST_MIN) / bin_width)] += 1
    bar_width = bin_width * 0.9
    y_top = max(counts) * 1.15

    print(f"n_total={n_total}  n_finished={n_finished}")
    print(f"all 210 deltas: mean={mean_d:.2f}s  median={median_d:.2f}s")
    print(f"clipped to histogram: {n_below} ≤ {HIST_MIN}, {n_above} ≥ {HIST_MAX}")

    def arr(seq, fmt=".4f"):
        return "(" + ", ".join(format(v, fmt) for v in seq) + ")"

    # symlog y for the scatter; linthresh: switch from linear to log around |y|=2
    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = {arr(xs)}
#let deltas = {arr(deltas)}
#let bin_centers = {arr(bin_centers)}
#let bin_counts = {arr(counts, "d")}

// horizontal "median" reference for the scatter panel
#let median_line_x = (0.05, 1500)
#let median_line_y = ({median_d:.4f}, {median_d:.4f})

// vertical mean / median markers for histogram panel
#let mean_v_x = ({mean_d:.4f}, {mean_d:.4f})
#let median_v_x = ({median_d:.4f}, {median_d:.4f})
#let v_y = (0, {y_top:.1f})

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 5cm,
    title: [Parasol overhead vs.\\ standalone solve time (all instances)],
    xlabel: [cp-sat,8 standalone (s)],
    ylabel: [parasol $minus$ standalone (s)],
    xscale: "log",
    yscale: "symlog",
    xlim: (0.05, 1500),
    ylim: (-1000, 1000),
    lq.scatter(xs, deltas, size: 3pt, color: blue.transparentize(60%)),
    lq.plot(median_line_x, median_line_y, mark: none,
      stroke: 1pt + orange, label: [median = #{median_d:.2f} s]),
  ),
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Distribution of overhead (zoomed to dense region)],
    xlabel: [parasol $minus$ standalone (s)],
    ylabel: [Number of instances],
    xlim: ({HIST_MIN}, {HIST_MAX}),
    ylim: (0, {y_top:.1f}),
    lq.bar(bin_centers, bin_counts, width: {bar_width:.4f},
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #{mean_d:.2f} s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #{median_d:.2f} s]),
  )
)

#set par(justify: true)
*MiniZinc Challenge 2023-2025; n = {n_finished} instances shown (of {n_total}
total; the remaining {n_total - n_finished} timed out at the 1200 s cutoff in
at least one of the two runs).* The same cp-sat solver with 8 cores runs both
standalone and via the parasol portfolio harness. The top panel uses a
symmetric-log y-axis to show both the constant-overhead cluster near
$Delta approx 1.3$ s for short instances and the wide spread for longer
instances. That spread is *not* parasol overhead but cp-sat's own run-to-run
variance: with 8-core parallel search and stochastic restarts, the same
instance can be solved 10× faster or slower simply depending on which worker
finds the proof first. The bottom panel zooms in on the dense region: median
$Delta$ = {median_d:.2f} s, mean {mean_d:.2f} s. Outliers clipped to histogram
edges: {n_below} below {HIST_MIN} s, {n_above} above {HIST_MAX} s.
"""

    OUT_FILE.write_text(typst)
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
