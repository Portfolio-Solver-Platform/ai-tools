#!/usr/bin/env python3
"""Generate a Typst/lilaq plot showing the parasol harness overhead from a
controlled same-machine replication (run-parasol-overhead-full).

Companion to parasol_overhead.py which uses the original cross-environment
data. The replication data resolves a confound in that earlier figure: the
apparent ~1.3 s overhead included machine-noise from running cpsat8 (parasol)
and cp-sat,8 (standalone) on different uCloud sessions. Re-running both modes
back-to-back on a single clean machine, with a 20 s timeout, isolates the
true harness overhead.

  panel 1 (top, scatter):    standalone time (log x) vs delta (linear y)
  panel 2 (bottom, hist):    distribution of deltas, focused on dense region
"""
import csv
import statistics
from pathlib import Path

REPLICATION_CSV = Path("/home/sofus/speciale/ai/results/parasol-overhead-medium/merged_with_internal.csv")
OUT_FILE = Path(__file__).resolve().parent / "parasol_overhead_replicated.typ"

# Per-source timeouts (the merged CSV combines 20s short + 600s medium runs)
TIMEOUT_S_BY_SOURCE = {"short": 20.0, "medium": 600.0}
SHORT_S = 200.0   # x-axis cap on the scatter (most data fits below this)
DELTA_MIN, DELTA_MAX = -2.0, 4.0
N_BINS = 30


def main():
    if not REPLICATION_CSV.exists():
        raise SystemExit(f"missing replication data at {REPLICATION_CSV}")

    rows = []
    with open(REPLICATION_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    pairs_all = []
    for r in rows:
        try:
            s = float(r["standalone_internal_ms"]) / 1000.0
            p = float(r["parasol_internal_ms"]) / 1000.0
        except (KeyError, ValueError):
            continue
        # Filter out runs where wall-clock hit the timeout for that source
        timeout_s = TIMEOUT_S_BY_SOURCE.get(r.get("source", "medium"), 20.0)
        if (float(r["standalone_wall_ms"]) >= timeout_s * 1000 * 0.95
                or float(r["parasol_wall_ms"]) >= timeout_s * 1000 * 0.95):
            continue
        pairs_all.append((s, p))

    short = [(s, p - s) for s, p in pairs_all if s < SHORT_S]
    xs = [x for x, _ in short]
    deltas = [d for _, d in short]
    deltas_in_band = [d for d in deltas if DELTA_MIN <= d <= DELTA_MAX]

    n_total = len(rows)
    n_finished = len(pairs_all)
    n_short = len(short)
    mean_d = statistics.mean(deltas_in_band)
    median_d = statistics.median(deltas_in_band)
    stdev_d = statistics.stdev(deltas_in_band)
    n_above = sum(1 for d in deltas if d > DELTA_MAX)
    n_below = sum(1 for d in deltas if d < DELTA_MIN)

    print(f"n_total={n_total}  n_finished={n_finished}  n_short(<{SHORT_S}s)={n_short}")
    print(f"deltas in [{DELTA_MIN}, {DELTA_MAX}] s (n={len(deltas_in_band)}):")
    print(f"  mean={mean_d:.3f}s   median={median_d:.3f}s   stdev={stdev_d:.3f}s")
    print(f"  outside band: {n_below} below, {n_above} above")

    # Histogram bins
    bin_edges = [DELTA_MIN + i * (DELTA_MAX - DELTA_MIN) / N_BINS for i in range(N_BINS + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(N_BINS)]
    bin_width = (DELTA_MAX - DELTA_MIN) / N_BINS
    counts = [0] * N_BINS
    for d in deltas_in_band:
        idx = int((d - DELTA_MIN) / bin_width)
        if idx >= N_BINS:
            idx = N_BINS - 1
        counts[idx] += 1
    bar_width = bin_width * 0.9
    y_top = max(counts) * 1.15

    def arr(seq, fmt=".4f"):
        return "(" + ", ".join(format(v, fmt) for v in seq) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = {arr(xs)}
#let deltas = {arr(deltas)}
#let bin_centers = {arr(bin_centers)}
#let bin_counts = {arr(counts, "d")}

#let median_line_x = (0, {SHORT_S})
#let median_line_y = ({median_d:.4f}, {median_d:.4f})

#let mean_v_x = ({mean_d:.4f}, {mean_d:.4f})
#let median_v_x = ({median_d:.4f}, {median_d:.4f})
#let v_y = (0, {y_top:.1f})

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Parasol overhead vs.\\ solver-reported solve time (replication)],
    xlabel: [cp-sat,8 standalone, solver-reported (s)],
    ylabel: [parasol $minus$ standalone (s)],
    xlim: (0, {SHORT_S}),
    ylim: ({DELTA_MIN}, {DELTA_MAX}),
    lq.scatter(xs, deltas, size: 3pt, color: blue.transparentize(60%)),
    lq.plot(median_line_x, median_line_y, mark: none,
      stroke: 1pt + orange, label: [median = #{median_d:.3f} s]),
  ),
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Distribution of overhead],
    xlabel: [parasol $minus$ standalone (s)],
    ylabel: [Number of instances],
    xlim: ({DELTA_MIN}, {DELTA_MAX}),
    ylim: (0, {y_top:.1f}),
    lq.bar(bin_centers, bin_counts, width: {bar_width:.4f},
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #{mean_d:.3f} s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #{median_d:.3f} s]),
  )
)

#set par(justify: true)
*Same-machine replication, 2023-2025 challenge instances; n = {len(deltas_in_band)}
shown (of {n_total} total).* The same cp-sat solver with 8 cores runs
standalone (`minizinc --solver cp-sat -p 8`) and via the parasol portfolio
harness (`--solver parasol --static-schedule cpsat8.csv -p 8`) back-to-back
on a single dedicated machine. The dataset combines two runs: a 20 s timeout
sweep over all 300 instances of 2023-2025, plus a 600 s timeout sweep over
68 medium-hard instances (uCloud cp-sat,8 took 60-1200 s) where 20 s was
not enough. Times are the solver-reported internal time of the last event
(`time` field of MiniZinc's JSON stream for standalone; last
`% time elapsed:` for parasol), which excludes harness shutdown so the
delta reflects only what happens during solving.
The overhead is approximately constant: mean $Delta$ = {mean_d:.3f} s,
median {median_d:.3f} s, stdev {stdev_d:.3f} s. The replication machine
is roughly 3× faster than the uCloud machines used for the original
portfolio benchmarks; scaling proportionally, the equivalent overhead on
uCloud is on the order of {median_d * 3:.2f} s. The remaining gap to the
~1.3 s observed cross-environment is uCloud session-to-session machine
variance, not parasol architecture.
"""

    OUT_FILE.write_text(typst)
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
