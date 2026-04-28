#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let xs = (0.3500, 33.4330, 26.3200, 40.4590, 1.8580, 0.3340, 0.1570, 57.0350, 0.7560, 1.0860, 0.9600, 0.9090, 0.9250, 22.8580, 0.0930, 3.0070, 2.3810, 0.1110, 7.5160, 65.5700, 14.7100, 3.4970, 3.2550, 9.3240, 24.3720, 1.0250, 71.5330, 19.5550, 0.1390, 167.0940, 0.1170, 0.0700, 0.1520, 0.1030, 0.3190, 0.7110, 1.5980, 5.2000, 66.2590, 0.1900, 1.5340, 1.4370, 24.0420, 24.5000, 0.5030, 5.9110, 3.3270, 3.9370, 15.5640, 0.6480, 0.8340, 2.1040, 0.2290, 1.0570, 1.0000, 2.1120, 6.6430, 13.0410, 59.8280, 0.2140, 0.2380, 5.3810, 6.0370, 81.4730, 80.6540, 11.0220, 3.6420, 4.4520, 0.2150, 11.0320, 11.7760, 13.0040, 2.6380, 2.2730, 2.2450, 1.6810, 2.2360, 0.3150, 0.8790, 13.7910, 15.3850, 0.6850, 1.6710, 0.1350, 0.1850, 0.5470, 0.1440, 0.2000, 0.1450, 0.5460, 0.2110, 0.1690, 0.7410, 0.5740, 0.6180, 0.1710, 1.6840, 0.1920, 0.2030, 0.1150, 0.1950, 0.1560, 0.4460, 0.3510, 0.3210, 1.6150, 0.1600, 0.2480, 4.6780, 0.2410, 5.0120, 0.9820, 0.0820, 0.1760, 0.1660, 0.6760, 0.1040, 0.1260, 0.1750, 0.7390, 0.4410, 17.3090, 5.5390, 13.3590, 0.6670, 3.0400, 0.0630, 0.0770, 1.4990, 11.1520, 14.4300, 0.3190, 2.0340, 2.7000, 4.5220, 3.6380, 13.0100, 2.4760, 4.3000, 0.4470, 5.0280, 3.7460, 7.5900, 17.9440, 4.3350, 15.0320, 13.7990, 0.3850, 1.1810, 0.2740, 1.1580, 0.3380, 0.1090, 0.1550, 0.1040, 0.2490, 0.3780, 1.0070, 0.7520, 4.1910, 10.4830, 0.9260, 3.0530, 3.2240, 3.3830, 0.5760, 1.3170, 0.9680, 0.4670, 0.5220, 0.4630, 13.2800, 0.0650, 1.1670, 7.7010, 15.3470, 6.8570, 1.4680, 0.3360, 2.6110, 1.1540, 0.1040, 0.5840, 0.2820, 4.2340, 12.8010)
#let deltas = (0.1270, -2.4420, -0.0020, -3.5350, 0.6540, 0.2190, 0.0060, 43.5910, 0.3350, 0.2960, 0.1090, 0.2810, 0.1600, 2.1510, 0.1050, -0.5570, 0.5270, 0.1120, 0.2480, 10.7280, 0.1580, 0.3890, 0.5100, 0.4760, 5.4290, 0.4420, 20.4820, -2.9740, 0.0970, 52.8310, 0.0850, 0.0930, 0.1470, 0.1040, 0.1730, 0.1410, 0.3490, -0.5000, -1.6890, 0.1100, 0.6810, 0.6390, 3.8680, 2.1500, 0.4020, 0.6440, 1.0120, 0.8610, 4.7230, 0.6270, 0.1170, -0.6390, 0.1460, 0.3240, 0.5950, 0.4770, 0.3760, -0.3250, 69.2720, 0.0960, 0.1020, -0.4380, -0.9440, 85.3370, -6.6720, -0.2060, -0.0230, 0.0690, 0.0800, -0.8450, -1.3360, -1.3220, -0.3920, 0.0240, -0.1600, 0.4500, -0.3610, 0.1680, 0.3410, -1.0920, -1.1230, 0.2320, -0.0240, 0.1900, 0.1540, 0.0950, 0.1730, 0.1520, 0.1490, 0.1280, 0.1700, 0.1430, -0.0790, 0.1170, -0.0040, 0.1320, -0.1170, 0.1460, 0.1580, 0.1760, 0.1540, 0.1340, -0.0470, 0.1030, 0.2030, -0.0800, 0.1550, 0.1180, 1.1870, 0.0510, -0.8840, 0.5900, 0.1450, 0.1500, 0.1600, -0.2170, 0.1880, 0.2120, 0.1790, 0.3190, 0.1740, -4.0060, 2.1940, 1.9400, 0.2780, 1.6200, 0.1050, 0.1180, 0.6520, 0.8200, 1.4640, 0.2430, 0.5530, 0.9010, 0.4690, 0.9000, 0.6710, 0.0240, 2.2650, 1.0900, 0.8700, 1.9140, 3.0950, -1.5620, -0.0420, 1.8540, 1.6290, 0.1520, -0.0810, 0.1310, 0.1970, 0.0970, 0.1410, 0.1230, 0.2160, 0.1300, 0.3840, 0.0120, 0.0370, 0.4540, -2.1840, 0.2320, -0.4480, 0.5550, 0.6150, 0.2700, 0.4330, 0.4240, 0.3540, 0.3970, 0.2930, 5.0160, 0.0920, -0.2760, 2.7970, 0.8520, 0.8810, 0.1450, 0.1390, 0.1170, 0.0630, 0.1530, 0.1160, 0.1750, 0.4620, -0.3870)
#let bin_centers = (-1.9000, -1.7000, -1.5000, -1.3000, -1.1000, -0.9000, -0.7000, -0.5000, -0.3000, -0.1000, 0.1000, 0.3000, 0.5000, 0.7000, 0.9000, 1.1000, 1.3000, 1.5000, 1.7000, 1.9000, 2.1000, 2.3000, 2.5000, 2.7000, 2.9000, 3.1000, 3.3000, 3.5000, 3.7000, 3.9000)
#let bin_counts = (0, 1, 1, 2, 2, 3, 1, 4, 7, 11, 69, 23, 16, 8, 7, 3, 0, 1, 2, 3, 3, 1, 0, 1, 0, 1, 0, 0, 0, 1)

#let median_line_x = (0, 200.0)
#let median_line_y = (0.1550, 0.1550)

#let mean_v_x = (0.2953, 0.2953)
#let median_v_x = (0.1550, 0.1550)
#let v_y = (0, 79.3)

#stack(
  dir: ttb, spacing: 0.6em,
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Parasol overhead vs.\ solver-reported solve time (replication)],
    xlabel: [cp-sat,8 standalone, solver-reported (s)],
    ylabel: [parasol $minus$ standalone (s)],
    xlim: (0, 200.0),
    ylim: (-2.0, 4.0),
    lq.scatter(xs, deltas, size: 3pt, color: blue.transparentize(60%)),
    lq.plot(median_line_x, median_line_y, mark: none,
      stroke: 1pt + orange, label: [median = #0.155 s]),
  ),
  lq.diagram(
    width: 9cm, height: 4.5cm,
    title: [Distribution of overhead],
    xlabel: [parasol $minus$ standalone (s)],
    ylabel: [Number of instances],
    xlim: (-2.0, 4.0),
    ylim: (0, 79.3),
    lq.bar(bin_centers, bin_counts, width: 0.1800,
      fill: blue.lighten(30%)),
    lq.plot(mean_v_x, v_y, mark: none, stroke: 1pt + red,
      label: [mean = #0.295 s]),
    lq.plot(median_v_x, v_y, mark: none, stroke: 1pt + orange,
      label: [median = #0.155 s]),
  )
)

#set par(justify: true)
*Same-machine replication, 2023-2025 challenge instances; n = 171
shown (of 300 total).* The same cp-sat solver with 8 cores runs
standalone (`minizinc --solver cp-sat -p 8`) and via the parasol portfolio
harness (`--solver parasol --static-schedule cpsat8.csv -p 8`) back-to-back
on a single dedicated machine. The dataset combines two runs: a 20 s timeout
sweep over all 300 instances of 2023-2025, plus a 600 s timeout sweep over
68 medium-hard instances (uCloud cp-sat,8 took 60-1200 s) where 20 s was
not enough. Times are the solver-reported internal time of the last event
(`time` field of MiniZinc's JSON stream for standalone; last
`% time elapsed:` for parasol), which excludes harness shutdown so the
delta reflects only what happens during solving.
The overhead is approximately constant: mean $Delta$ = 0.295 s,
median 0.155 s, stdev 0.728 s. The replication machine
is roughly 3× faster than the uCloud machines used for the original
portfolio benchmarks; scaling proportionally, the equivalent overhead on
uCloud is on the order of 0.46 s. The remaining gap to the
~1.3 s observed cross-environment is uCloud session-to-session machine
variance, not parasol architecture.
