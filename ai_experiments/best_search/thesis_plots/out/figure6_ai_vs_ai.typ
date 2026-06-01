#figure(
  {
    let n = 15
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 16cm, height: 7cm,
      title: [Per-year 3-way Borda: AI-k1 vs AI-ek1 deployment],
      xlabel: [Year], ylabel: [Borda (3-way tournament)],
      xaxis: (ticks: ((0, [2011]), (1, [2012]), (2, [2013]), (3, [2014]), (4, [2015]), (5, [2016]), (6, [2017]), (7, [2018]), (8, [2019]), (9, [2020]), (10, [2021]), (11, [2022]), (12, [2023]), (13, [2024]), (14, [2025])), subticks: none),
      ylim: (75, 146),
      lq.bar(xs.map(x => x - 0.3), (115.58, 112.26, 105.14, 130.79, 101.90, 119.30, 102.80, 118.77, 112.55, 105.52, 108.02, 111.33, 111.34, 101.84, 80.98), width: 0.2, fill: gray.lighten(30%), label: [always-cpsat]),
      lq.bar(xs.map(x => x - 0.1), (125.49, 115.32, 119.02, 130.79, 111.14, 123.71, 110.38, 120.40, 116.57, 114.21, 107.92, 116.93, 112.35, 108.48, 89.04), width: 0.2, fill: blue.lighten(20%), label: [AI-k1 deployment]),
      lq.bar(xs.map(x => x + 0.1), (117.74, 112.26, 114.33, 131.79, 110.94, 119.30, 111.54, 118.77, 115.04, 107.55, 109.02, 112.46, 111.34, 106.31, 82.78), width: 0.2, fill: orange.lighten(20%), label: [AI-ek1 deployment]),
      lq.bar(xs.map(x => x + 0.3), (134.03, 132.39, 135.65, 138.64, 127.58, 127.38, 136.27, 129.42, 126.08, 124.68, 126.56, 133.08, 129.76, 120.77, 115.20), width: 0.2, fill: green.lighten(50%), label: [Oracle (best of 3)]),
    )
  },
  caption: [Per-year LOYO 3-way Borda comparison of the two AI deployments. Each year is held out; both AIs are independently fit on the other 14 years and evaluated on the held-out year's instances. Each AI's pick (cpsat vs its alternative) is scored in a 3-way pairwise tournament against {cpsat, k1, ek1}, so the two deployment strategies are directly comparable on the same scale. always-cpsat and oracle (per-instance best of all 3) bound the achievable range.],
) <fig:ai-vs-ai>
