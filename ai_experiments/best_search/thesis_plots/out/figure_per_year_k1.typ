#figure(
  {
    let n = 15
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 16cm, height: 7cm,
      title: [Per-year LOYO Borda, cpsat8 vs k1],
      xlabel: [Year], ylabel: [Borda score],
      xaxis: (ticks: ((0, [2011]), (1, [2012]), (2, [2013]), (3, [2014]), (4, [2015]), (5, [2016]), (6, [2017]), (7, [2018]), (8, [2019]), (9, [2020]), (10, [2021]), (11, [2022]), (12, [2023]), (13, [2024]), (14, [2025])), subticks: none),
      ylim: (28, 76),
      lq.bar(xs.map(x => x - 0.32), (54.94, 54.50, 51.49, 63.82, 51.77, 55.98, 50.91, 56.77, 53.69, 49.62, 52.69, 54.28, 51.97, 49.34, 37.11), width: 0.15, fill: gray.lighten(20%), label: [always-cpsat]),
      lq.bar(xs.map(x => x - 0.16), (45.06, 45.50, 43.51, 34.18, 38.23, 39.02, 39.09, 33.23, 35.31, 35.38, 35.31, 37.72, 38.03, 34.66, 39.89), width: 0.15, fill: purple.lighten(30%), label: [always-k1]),
      lq.bar(xs.map(x => x), (61.43, 55.34, 59.39, 62.18, 55.80, 57.49, 54.62, 57.73, 55.35, 54.73, 56.70, 56.64, 52.04, 51.46, 42.19), width: 0.15, fill: red.lighten(30%), label: [Plain SVC]),
      lq.bar(xs.map(x => x + 0.16), (61.32, 56.12, 60.86, 63.95, 57.55, 58.80, 54.98, 58.03, 57.24, 55.78, 53.47, 58.00, 53.22, 53.67, 41.51), width: 0.15, fill: blue.lighten(10%), label: [BagSVC-MW]),
      lq.bar(xs.map(x => x + 0.32), (66.59, 66.59, 68.87, 67.57, 65.53, 61.63, 67.28, 63.24, 62.53, 61.61, 64.91, 66.40, 63.33, 61.51, 59.02), width: 0.15, fill: green.lighten(30%), label: [Oracle]),
    )
  },
  caption: [Per-year LOYO Borda on cpsat8 vs k1. Each year is held out from training; the model is fit on the other 14 and evaluated on the held-out year. Bars are: always-cpsat and always-k1 (the two no-AI baselines; they sum to the year's instance count when every instance is solved by at least one portfolio), Plain SVC (single RBF SVC), BagSVC-MW (this work), and Oracle (per-instance best portfolio). Note 2025 is the only year where always-k1 beats always-cpsat. Higher is better.],
) <fig:k1>
