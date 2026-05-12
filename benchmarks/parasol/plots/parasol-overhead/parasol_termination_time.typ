#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs_clean = (0.4770, 30.9910, 26.3180, 36.9240, 2.5120, 0.5530, 0.1630, 100.6260, 1.0910, 1.3820, 1.0690, 1.1900, 1.0850, 25.0090, 0.1980, 2.4500, 2.9080, 0.2230, 7.7640, 76.2980, 14.8680, 3.8860, 3.7650, 9.8000, 29.8010, 1.4670, 92.0150, 16.5810, 0.2360, 219.9250, 0.2020, 0.1630, 0.2990, 0.2070, 0.4920, 0.8520, 1.9470, 4.7000, 64.5700, 0.3000, 2.2150, 2.0760, 27.9100, 26.6500, 0.9050, 6.5550, 4.3390, 4.7980, 20.2870, 1.2750, 0.9510, 1.4650, 0.3750, 1.3810, 1.5950, 2.5890, 7.0190, 12.7160, 129.1000, 0.3100, 0.3400, 4.9430, 5.0930, 166.8100, 73.9820, 10.8160, 3.6190)
#let ys_clean = (0.0060, 0.0090, 0.0070, 0.0060, 0.0230, 0.0060, 0.0090, 0.0060, 0.0050, 0.0220, 0.0060, 0.0060, 0.0060, 0.0060, 0.0040, 0.0140, 0.0200, 0.0050, 0.0230, 0.0070, 0.0080, 0.0170, 0.0260, 0.0220, 0.0070, 0.0190, 0.0060, 0.0060, 0.0050, 0.0060, 0.0060, 0.0050, 0.0040, 0.0050, 0.0040, 0.0240, 0.0330, 0.0170, 0.0070, 0.0050, 0.0160, 0.0210, 0.0090, 0.0090, 0.0060, 0.0200, 0.0220, 0.0280, 0.0080, 0.0170, 0.0190, 0.0250, 0.0050, 0.0190, 0.0240, 0.0140, 0.0280, 0.0060, 0.0060, 0.0050, 0.0050, 0.0220, 0.0310, 0.0080, 0.0070, 0.0060, 0.0240)
#let xs_to    = (16.0780,)
#let ys_to    = (583.9280,)
#let bin_centers = (-0.4750, -0.4250, -0.3750, -0.3250, -0.2750, -0.2250, -0.1750, -0.1250, -0.0750, -0.0250, 0.0250, 0.0750, 0.1250, 0.1750, 0.2250, 0.2750, 0.3250, 0.3750, 0.4250, 0.4750, 0.5250, 0.5750, 0.6250, 0.6750, 0.7250, 0.7750, 0.8250, 0.8750, 0.9250, 0.9750, 1.0250, 1.0750, 1.1250, 1.1750, 1.2250, 1.2750, 1.3250, 1.3750, 1.4250, 1.4750)
#let bin_counts  = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

#let median_term = 0.0080
#let mean_term   = 8.5995
#let v_y = (0, 77.0)
#let median_v_x = (median_term, median_term)
#let mean_v_x   = (mean_term, mean_term)

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Parasol termination time vs.\ solver-reported solve time],
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
    xlim: (-0.5, 1.5),
    ylim: (0, 77.0),
    lq.bar(bin_centers, bin_counts, width: 0.0450,
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #8.60 s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #0.008 s]),
  )
)

#set par(justify: true)
*Same-machine replication, 2025 challenge instances; n = 68 pairs.*
We pick the 68 instances where uCloud cp-sat,8 finished in 60-1200 s,
re-run them on a clean machine with a 600 s wall-clock cap, and look at
the gap between the solver's last reported event (last `% time elapsed:`
for parasol) and the process exiting. For runs that finished with an
Optimal or Unsat proof this is just harness shutdown — median
0.008 s
across cleanly-finished instances (67/68).
The single timeout-hit instance (red) is `skill-allocation/skill_allocation_mzn_2m_2`:
parasol's last solver event was at 16.1 s,
yet the process kept running until SIGTERM fired at 600 s — almost ten
minutes of doing nothing. Standalone cp-sat shuts down much tighter
(median 0.018 s, max 0.563 s),
indicating that parasol does not always notice when its workers have
stopped making progress.
