#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = (0.0730, 0.0850, 0.1220, 0.1250, 0.1450, 0.1510, 0.1880, 0.2450, 0.2510, 0.2550, 0.2890, 0.2980, 0.4150, 0.4370, 0.5730, 0.8270, 0.8660, 0.9730, 1.0100, 1.2260, 1.2310, 1.2530, 1.2600, 1.2910, 1.3210, 1.3910, 1.4500, 2.0740, 2.2210, 2.2500, 2.3500, 2.4140, 3.1760, 3.3790, 3.6290, 3.9950, 4.5260, 4.6720, 5.0630, 5.1350, 5.8800, 6.0240, 6.4750, 8.0780, 8.1990, 8.4730, 12.4430, 15.7610, 18.1480, 19.0430, 20.3230, 21.6590, 23.0360, 28.8020, 30.8180, 31.1360, 32.3290, 39.6630, 44.4490, 55.1300, 68.3250, 73.2320, 74.9810, 77.1120, 92.4620, 186.8530, 200.9700, 344.8710, 606.9990, 842.9500)
#let ds = (0.1110, 0.0940, 0.1500, 0.2030, 0.1580, 0.1520, 0.1230, 0.2050, 0.1240, 0.1850, 0.1950, 0.1520, 0.2660, 0.1270, 0.0840, 0.2700, 0.3890, 0.4400, 0.0830, 0.0200, 0.1090, 0.0280, 0.0890, 0.1010, 0.2450, 0.0040, 0.3040, 0.2030, 0.8380, -0.6590, 0.5470, -0.0200, -0.6820, -0.1550, -0.6890, -0.2340, 0.5350, 0.6790, 0.0000, 0.2460, 0.6760, 0.1360, 1.1120, -0.2650, 0.2120, 0.7550, -0.7860, -0.4010, 0.7790, 1.5770, 2.3140, 4.7660, 0.9560, 1.7100, 4.5020, 2.2800, -11.5020, -0.6520, -1.1210, 8.5830, -5.5170, 5.6380, 449.8870, 6.9530, -9.0810, 41.6260, 1.4640, -84.7430, -56.4580, 92.5110)
#let elo = (0.0170, 0.0120, 0.0540, 0.0900, 0.0330, 0.0150, 0.0720, 0.0840, 0.0380, 0.0270, 0.0460, 0.0780, 0.0930, 0.0800, 0.1000, 0.3600, 0.1050, 0.1860, 0.0440, 0.6000, 0.1070, 0.1190, 0.1070, 0.3210, 0.1060, 0.1030, 0.1800, 0.2000, 0.2990, 0.3540, 0.0800, 0.2950, 0.0550, 0.1840, 1.2950, 1.8800, 1.9450, 0.5280, 0.8400, 0.2910, 0.6330, 0.4630, 0.4090, 0.5600, 1.1710, 1.2690, 1.3680, 0.8010, 1.1460, 4.2570, 7.1980, 1.6060, 1.6050, 9.5960, 6.1200, 6.4020, 0.6050, 0.9000, 3.6930, 30.2530, 4.6560, 8.3160, 398.8250, 3.7630, 19.6960, 183.9680, 78.8340, 874.7930, 40.2140, 321.6620)
#let ehi = (0.0060, 0.0480, 0.0360, 0.0610, 0.0610, 0.0500, 0.0970, 0.0900, 0.0880, 0.0320, 0.0750, 0.0570, 0.0940, 0.2190, 0.2080, 0.1580, 0.1980, 0.0320, 0.1330, 0.5250, 0.1690, 0.1210, 0.1150, 0.1870, 0.1890, 0.0450, 0.0740, 0.0660, 0.1930, 1.1670, 0.5390, 1.6720, 1.0060, 0.1090, 1.0020, 1.3820, 0.8120, 0.6160, 1.0210, 1.5140, 0.9580, 0.8250, 0.7390, 0.7540, 0.8890, 0.0970, 3.0080, 1.4170, 0.3910, 0.7690, 6.6470, 0.6750, 1.9540, 6.3690, 1.1090, 2.1800, 14.3800, 1.4210, 5.9420, 74.3940, 29.2810, 7.5000, 42.5680, 5.8840, 33.9100, 92.2620, 22.8650, 203.5420, 81.5210, 96.5070)
#let bin_centers = (-3.8667, -3.6000, -3.3333, -3.0667, -2.8000, -2.5333, -2.2667, -2.0000, -1.7333, -1.4667, -1.2000, -0.9333, -0.6667, -0.4000, -0.1333, 0.1333, 0.4000, 0.6667, 0.9333, 1.2000, 1.4667, 1.7333, 2.0000, 2.2667, 2.5333, 2.8000, 3.0667, 3.3333, 3.6000, 3.8667)
#let bin_counts = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 5, 1, 4, 28, 4, 6, 2, 1, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0)

#let median_line_x = (0.05, 1200)
#let median_line_y = (0.1715, 0.1715)

#let mean_v_x = (6.6133, 6.6133)
#let median_v_x = (0.1715, 0.1715)
#let v_y = (0, 32.2)

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 5cm,
    title: [Parasol overhead, 3-rep replication (2025, $t_max$ = 1200 s, internal time)],
    xlabel: [cp-sat,8 standalone internal median (s)],
    ylabel: [median(parasol) $minus$ median(standalone), internal (s)],
    xscale: "log",
    xlim: (0.05, 1200),
    ylim: (-4.0, 4.0),
    legend: (position: top + left),
    lq.plot(xs, ds, yerr: (m: elo, p: ehi), mark: "o", mark-size: 3pt,
      stroke: none, color: blue.transparentize(50%)),
    lq.plot(median_line_x, median_line_y, mark: none,
      stroke: 1pt + orange, label: [median = #0.171 s]),
  ),
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Distribution of per-instance median deltas (internal time)],
    xlabel: [median(parasol) $minus$ median(standalone), internal (s)],
    ylabel: [Number of instances],
    xlim: (-4.0, 4.0),
    ylim: (0, 32.2),
    legend: (position: top + left),
    lq.bar(bin_centers, bin_counts, width: 0.2400,
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #6.613 s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #0.171 s]),
  )
)

#set par(justify: true)
*MiniZinc Challenge 2025; 3 reps per (mode, instance), n = 70 of
100 shown (the remaining 30 had at least one
rep that hit the 1200 s wall-clock timeout and were excluded
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
70 cleanly-finished pairs is 0.171 s
(mean 6.613 s, stdev 56.526 s). The within-mode noise floor
— median range across the 3 reps — is 0.472 s for standalone and
0.558 s for parasol, i.e. larger than the median overhead, so for
any single instance the delta you'd measure from one rep is dominated by
solver variance, not by parasol's per-invocation cost.
