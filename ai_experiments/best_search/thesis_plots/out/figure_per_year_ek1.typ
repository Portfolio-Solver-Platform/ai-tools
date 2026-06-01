#figure(
  {
    let n = 15
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 16cm, height: 7cm,
      title: [Per-year LOYO Borda, cpsat8 vs ek1],
      xlabel: [Year], ylabel: [Borda score],
      xaxis: (ticks: ((0, [2011]), (1, [2012]), (2, [2013]), (3, [2014]), (4, [2015]), (5, [2016]), (6, [2017]), (7, [2018]), (8, [2019]), (9, [2020]), (10, [2021]), (11, [2022]), (12, [2023]), (13, [2024]), (14, [2025])), subticks: none),
      ylim: (24, 79),
      lq.bar(xs.map(x => x - 0.32), (60.64, 57.76, 53.65, 66.98, 50.13, 63.32, 51.89, 62.00, 58.86, 55.89, 55.33, 57.05, 59.37, 52.50, 43.87), width: 0.15, fill: gray.lighten(20%), label: [always-cpsat]),
      lq.bar(xs.map(x => x - 0.16), (39.36, 42.24, 41.35, 31.02, 39.87, 31.68, 40.11, 28.00, 30.14, 29.11, 32.67, 34.95, 30.63, 31.50, 32.13), width: 0.15, fill: purple.lighten(30%), label: [always-ek1]),
      lq.bar(xs.map(x => x), (61.58, 57.76, 55.53, 64.93, 52.79, 62.81, 56.93, 63.00, 59.11, 56.13, 56.08, 58.60, 57.37, 54.36, 43.06), width: 0.15, fill: red.lighten(30%), label: [Plain SVC]),
      lq.bar(xs.map(x => x + 0.16), (62.07, 57.76, 59.98, 68.70, 58.10, 63.29, 56.12, 62.00, 60.53, 57.23, 56.08, 59.74, 58.95, 55.48, 44.82), width: 0.15, fill: blue.lighten(10%), label: [BagSVC-MW]),
      lq.bar(xs.map(x => x + 0.32), (66.16, 66.76, 67.74, 72.25, 63.63, 65.36, 68.14, 67.22, 64.37, 62.91, 66.10, 67.44, 67.01, 63.66, 57.33), width: 0.15, fill: green.lighten(30%), label: [Oracle]),
    )
  },
  caption: [Per-year LOYO Borda on cpsat8 vs ek1. Each year is held out from training; the model is fit on the other 14 and evaluated on the held-out year. Bars are: always-cpsat and always-ek1 (the two no-AI baselines; they sum to the year's instance count when every instance is solved by at least one portfolio), Plain SVC (single RBF SVC), BagSVC-MW (this work), and Oracle (per-instance best portfolio). Note 2025 is the only year where always-ek1 beats always-cpsat. Higher is better.],
) <fig:ek1>
