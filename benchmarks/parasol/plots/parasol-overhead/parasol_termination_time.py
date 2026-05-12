#!/usr/bin/env python3
"""Termination-time plot: how long the harness sat between the solver's last
reported event and the process actually exiting.

  panel 1 (top, scatter):    standalone solver-reported time (linear x)
                             vs termination_time = wall_ms - internal_ms (linear y)
  panel 2 (bottom, hist):    distribution of parasol termination times,
                             with standalone for reference

For instances that ran to completion (Optimal/Unsat) this is just the harness
shutdown overhead — usually <100 ms. For instances killed by the timeout
command, this can be substantial: the solver gave up well before the timeout
fired, and the process hung idle until SIGTERM.
"""
import csv
import statistics
from pathlib import Path

REPLICATION_CSV = Path("/home/sofus/speciale/ai/results/parasol-overhead-medium/merged_with_internal.csv")
OUT_FILE = Path(__file__).resolve().parent / "parasol_termination_time.typ"

TIMEOUT_S_BY_SOURCE = {"short": 20.0, "medium": 600.0}
SHORT_S = 600.0   # x-axis cap (everything fits)
TT_MIN, TT_MAX = -0.5, 1.5  # most cleanly-finished termination is 0-200ms
N_BINS = 40


def fnum(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    if not REPLICATION_CSV.exists():
        raise SystemExit(f"missing {REPLICATION_CSV}")

    rows = []
    with open(REPLICATION_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Build per-row records. Termination time = wall - internal.
    # Restrict to the medium-hard sweep (600 s timeout) — the older 20 s short
    # sweep was deliberately capped tight to measure overhead on small instances
    # and would conflate "parasol hung" with "20 s budget was insufficient".
    records = []
    for r in rows:
        if r.get("source", "medium") != "medium":
            continue
        sw = fnum(r.get("standalone_wall_ms"))
        sn = fnum(r.get("standalone_internal_ms"))
        pw = fnum(r.get("parasol_wall_ms"))
        pn = fnum(r.get("parasol_internal_ms"))
        if None in (sw, sn, pw, pn):
            continue
        timeout_ms = TIMEOUT_S_BY_SOURCE.get(r.get("source", "medium"), 20.0) * 1000
        records.append({
            "source": r.get("source", "medium"),
            "year": r.get("year", ""),
            "problem": r.get("problem", ""),
            "name": r.get("name", ""),
            "standalone_wall": sw / 1000.0,
            "standalone_internal": sn / 1000.0,
            "parasol_wall": pw / 1000.0,
            "parasol_internal": pn / 1000.0,
            "standalone_term": (sw - sn) / 1000.0,
            "parasol_term": (pw - pn) / 1000.0,
            "standalone_timed_out": sw >= timeout_ms * 0.95,
            "parasol_timed_out": pw >= timeout_ms * 0.95,
        })

    n_total = len(records)
    print(f"records: {n_total}")

    s_term = [r["standalone_term"] for r in records]
    p_term = [r["parasol_term"] for r in records]
    print(f"standalone termination (s): median={statistics.median(s_term):.3f}  mean={statistics.mean(s_term):.3f}  max={max(s_term):.3f}")
    print(f"parasol    termination (s): median={statistics.median(p_term):.3f}  mean={statistics.mean(p_term):.3f}  max={max(p_term):.3f}")

    # Scatter: x = standalone solver-reported time, y = parasol termination time.
    # Color/marker: distinguish parasol-timed-out from cleanly-finished.
    pts_clean = [(r["parasol_internal"], r["parasol_term"]) for r in records if not r["parasol_timed_out"]]
    pts_to    = [(r["parasol_internal"], r["parasol_term"]) for r in records if r["parasol_timed_out"]]

    # Hist: parasol termination time, capped to TT_MAX
    bin_edges = [TT_MIN + i * (TT_MAX - TT_MIN) / N_BINS for i in range(N_BINS + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(N_BINS)]
    bin_width = (TT_MAX - TT_MIN) / N_BINS
    counts = [0] * N_BINS
    for tt in p_term:
        idx = int((tt - TT_MIN) / bin_width)
        if idx < 0:
            counts[0] += 1
        elif idx >= N_BINS:
            counts[-1] += 1
        else:
            counts[idx] += 1
    bar_width = bin_width * 0.9
    y_top = max(counts) * 1.15

    def arr(seq, fmt=".4f"):
        items = [format(v, fmt) for v in seq]
        if len(items) == 1:
            return f"({items[0]},)"  # length-1 typst tuple needs trailing comma
        return "(" + ", ".join(items) + ")"

    typst = f"""\
#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs_clean = {arr([x for x, y in pts_clean])}
#let ys_clean = {arr([y for x, y in pts_clean])}
#let xs_to    = {arr([x for x, y in pts_to])}
#let ys_to    = {arr([y for x, y in pts_to])}
#let bin_centers = {arr(bin_centers)}
#let bin_counts  = {arr(counts, "d")}

#let median_term = {statistics.median(p_term):.4f}
#let mean_term   = {statistics.mean(p_term):.4f}
#let v_y = (0, {y_top:.1f})
#let median_v_x = (median_term, median_term)
#let mean_v_x   = (mean_term, mean_term)

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Parasol termination time vs.\\ solver-reported solve time],
    xlabel: [parasol solver-reported time (s)],
    ylabel: [wall $minus$ internal (s)],
    xlim: (0, 600),
    ylim: (-1, 600),
    lq.scatter(xs_clean, ys_clean, size: 3pt, color: blue.transparentize(60%),
      label: [finished cleanly]),
    lq.scatter(xs_to, ys_to, size: 4pt, color: red.transparentize(40%),
      label: [hit wall-clock timeout]),
  ),
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Distribution of parasol termination time],
    xlabel: [wall $minus$ internal (s)],
    ylabel: [Number of instances],
    xlim: ({TT_MIN}, {TT_MAX}),
    ylim: (0, {y_top:.1f}),
    lq.bar(bin_centers, bin_counts, width: {bar_width:.4f},
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #{statistics.mean(p_term):.2f} s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #{statistics.median(p_term):.3f} s]),
  )
)

#set par(justify: true)
*Same-machine replication, 2025 challenge instances; n = {n_total} pairs.*
We pick the 68 instances where uCloud cp-sat,8 finished in 60-1200 s,
re-run them on a clean machine with a 600 s wall-clock cap, and look at
the gap between the solver's last reported event (last `% time elapsed:`
for parasol) and the process exiting. For runs that finished with an
Optimal or Unsat proof this is just harness shutdown — median
{statistics.median([r['parasol_term'] for r in records if not r['parasol_timed_out']]):.3f} s
across cleanly-finished instances ({sum(1 for r in records if not r['parasol_timed_out'])}/{n_total}).
The single timeout-hit instance (red) is `skill-allocation/skill_allocation_mzn_2m_2`:
parasol's last solver event was at {[r['parasol_internal'] for r in records if r['parasol_timed_out']][0] if any(r['parasol_timed_out'] for r in records) else 0:.1f} s,
yet the process kept running until SIGTERM fired at 600 s — almost ten
minutes of doing nothing. Standalone cp-sat shuts down much tighter
(median {statistics.median(s_term):.3f} s, max {max(s_term):.3f} s),
indicating that parasol does not always notice when its workers have
stopped making progress.
"""
    OUT_FILE.write_text(typst)
    print(f"\nwrote {OUT_FILE}")


if __name__ == "__main__":
    main()
