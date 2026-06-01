#!/usr/bin/env python3
"""Generate a Typst/lilaq plot of parasol overhead from the 3-rep 2025 sweep.

Uses three back-to-back reps per (mode, instance) and the solver-reported
"internal" time (last MiniZinc JSON `time` field for standalone, last
`% time elapsed:` for parasol). Both internal times measure "process start
to final event before shutdown" and exclude the post-event shutdown tail,
which removes parasol's ~50ms shutdown bias.
"""
import csv
import statistics
from pathlib import Path

MERGED_CSV = Path("/home/sofus/speciale/ai/results/parasol-overhead-2025-3reps/merged.csv")
OUT_FILE = Path(__file__).resolve().parent / "parasol_overhead_3reps.typ"

TIMEOUT_S = 1200.0
TIMEOUT_EPSILON_MS = 1000.0  # match analyze.py: any rep within 1s of timeout = timed out
HIST_MIN, HIST_MAX = -4.0, 4.0
N_BINS = 30


def main():
    if not MERGED_CSV.exists():
        raise SystemExit(f"missing merged data at {MERGED_CSV}")

    def f_or_none(s):
        try: return float(s)
        except (ValueError, TypeError): return None

    points = []  # (s_med, d_med, d_lo, d_hi) all in seconds (internal time)
    deltas_all_s = []
    n_total = 0
    timeout_trip_ms = TIMEOUT_S * 1000.0 - TIMEOUT_EPSILON_MS
    with open(MERGED_CSV) as f:
        for r in csv.DictReader(f):
            n_total += 1
            # Filter on wall time: any rep that hit the `timeout` killer has a
            # truncated .out, so its internal time isn't trustworthy.
            s_wall_max = max(float(r[f"standalone_ms_{i}"]) for i in (1, 2, 3))
            p_wall_max = max(float(r[f"parasol_ms_{i}"])    for i in (1, 2, 3))
            if s_wall_max >= timeout_trip_ms or p_wall_max >= timeout_trip_ms:
                continue

            s_med = f_or_none(r["standalone_internal_median_ms"])
            p_med = f_or_none(r["parasol_internal_median_ms"])
            s_min = f_or_none(r["standalone_internal_min_ms"])
            s_max = f_or_none(r["standalone_internal_max_ms"])
            p_min = f_or_none(r["parasol_internal_min_ms"])
            p_max = f_or_none(r["parasol_internal_max_ms"])
            if None in (s_med, p_med, s_min, s_max, p_min, p_max):
                continue
            s_med /= 1000.0; p_med /= 1000.0
            s_min /= 1000.0; s_max /= 1000.0
            p_min /= 1000.0; p_max /= 1000.0
            d_med = p_med - s_med
            d_lo = p_min - s_max  # worst-case under observed rep ranges
            d_hi = p_max - s_min
            points.append((s_med, d_med, d_lo, d_hi))
            deltas_all_s.append(d_med)

    if not points:
        raise SystemExit("no usable points")

    median_d = statistics.median(deltas_all_s)
    mean_d   = statistics.mean(deltas_all_s)
    stdev_d  = statistics.stdev(deltas_all_s) if len(deltas_all_s) > 1 else 0.0
    n_above  = sum(1 for d in deltas_all_s if d > HIST_MAX)
    n_below  = sum(1 for d in deltas_all_s if d < HIST_MIN)
    deltas_in_band = [d for d in deltas_all_s if HIST_MIN <= d <= HIST_MAX]

    bin_edges = [HIST_MIN + i * (HIST_MAX - HIST_MIN) / N_BINS for i in range(N_BINS + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(N_BINS)]
    bin_width = (HIST_MAX - HIST_MIN) / N_BINS
    counts = [0] * N_BINS
    for d in deltas_in_band:
        idx = int((d - HIST_MIN) / bin_width)
        if idx >= N_BINS: idx = N_BINS - 1
        counts[idx] += 1
    bar_width = bin_width * 0.9
    y_top = max(counts) * 1.15

    # Within-mode noise floor (internal max - internal min across reps), median across instances
    ranges_s = []
    ranges_p = []
    with open(MERGED_CSV) as f:
        for r in csv.DictReader(f):
            s_wall_max = max(float(r[f"standalone_ms_{i}"]) for i in (1, 2, 3))
            p_wall_max = max(float(r[f"parasol_ms_{i}"])    for i in (1, 2, 3))
            if s_wall_max >= timeout_trip_ms or p_wall_max >= timeout_trip_ms:
                continue
            rs = f_or_none(r.get("standalone_internal_range_ms"))
            rp = f_or_none(r.get("parasol_internal_range_ms"))
            if rs is None or rp is None:
                continue
            ranges_s.append(rs / 1000.0)
            ranges_p.append(rp / 1000.0)
    noise_s = statistics.median(ranges_s)
    noise_p = statistics.median(ranges_p)

    print(f"n_total={n_total}  n_kept={len(points)}")
    print(f"per-instance median delta: mean={mean_d:.3f}s  median={median_d:.3f}s  "
          f"stdev={stdev_d:.3f}s")
    print(f"clipped to histogram: {n_below} below {HIST_MIN}s, {n_above} above {HIST_MAX}s")
    print(f"within-mode noise (median range over 3 reps): "
          f"standalone {noise_s:.3f}s, parasol {noise_p:.3f}s")

    points.sort(key=lambda t: t[0])
    xs   = [s for s, _, _, _ in points]
    ds   = [d for _, d, _, _ in points]
    elo  = [d - lo for (_, d, lo, _) in points]
    ehi  = [hi - d for (_, d, _, hi) in points]

    def arr(seq, fmt=".4f"):
        return "(" + ", ".join(format(v, fmt) for v in seq) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = {arr(xs)}
#let ds = {arr(ds)}
#let elo = {arr(elo)}
#let ehi = {arr(ehi)}
#let bin_centers = {arr(bin_centers)}
#let bin_counts = {arr(counts, "d")}

#let median_line_x = (0.05, 1200)
#let median_line_y = ({median_d:.4f}, {median_d:.4f})

#let mean_v_x = ({mean_d:.4f}, {mean_d:.4f})
#let median_v_x = ({median_d:.4f}, {median_d:.4f})
#let v_y = (0, {y_top:.1f})

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 5cm,
    title: [Parasol overhead, 3-rep replication (2025, $t_max$ = 1200 s, internal time)],
    xlabel: [cp-sat,8 standalone internal median (s)],
    ylabel: [median(parasol) $minus$ median(standalone), internal (s)],
    xscale: "log",
    xlim: (0.05, 1200),
    ylim: ({HIST_MIN}, {HIST_MAX}),
    legend: (position: top + left),
    lq.plot(xs, ds, yerr: (m: elo, p: ehi), mark: "o", mark-size: 3pt,
      stroke: none, color: blue.transparentize(50%)),
    lq.plot(median_line_x, median_line_y, mark: none,
      stroke: 1pt + orange, label: [median = #{median_d:.3f} s]),
  ),
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Distribution of per-instance median deltas (internal time)],
    xlabel: [median(parasol) $minus$ median(standalone), internal (s)],
    ylabel: [Number of instances],
    xlim: ({HIST_MIN}, {HIST_MAX}),
    ylim: (0, {y_top:.1f}),
    legend: (position: top + left),
    lq.bar(bin_centers, bin_counts, width: {bar_width:.4f},
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #{mean_d:.3f} s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #{median_d:.3f} s]),
  )
)

#set par(justify: true)
*MiniZinc Challenge 2025; 3 reps per (mode, instance), n = {len(points)} of
{n_total} shown (the remaining {n_total - len(points)} had at least one
rep that hit the {TIMEOUT_S:.0f} s wall-clock timeout and were excluded
so the parsed internal times aren't truncated).*
Times shown are solver-reported "internal" time — the last
MiniZinc-emitted `time` field for standalone and the last
`% time elapsed:` line for parasol — both measured from process start to
the final event before shutdown. That excludes parasol's post-event
shutdown tail (median ≈ 50 ms in this sweep), so the delta isolates pure
orchestration overhead rather than total wallclock. Each point is one
instance: x = median standalone-internal time over 3 reps; y =
median(parasol) $minus$ median(standalone). Error bars span the worst-case
delta given the observed per-mode rep ranges
(min-parasol $minus$ max-standalone, max-parasol $minus$ min-standalone),
so their size reflects cp-sat's own run-to-run variance with 8-core
parallel search and stochastic restarts. Median overhead across these
{len(points)} cleanly-finished pairs is {median_d:.3f} s
(mean {mean_d:.3f} s, stdev {stdev_d:.3f} s). The within-mode noise floor
— median range across the 3 reps — is {noise_s:.3f} s for standalone and
{noise_p:.3f} s for parasol, i.e. larger than the median overhead, so for
any single instance the delta you'd measure from one rep is dominated by
solver variance, not by parasol's per-invocation cost.
"""

    OUT_FILE.write_text(typst)
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
