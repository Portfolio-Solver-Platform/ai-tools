#figure(
  {
    let n = 15
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 16cm, height: 7cm,
      title: [Year-by-year instance leakage under leave-one-year-out],
      xlabel: [Held-out year],
      ylabel: [Share of held-out instances also in training (%)],
      xaxis: (ticks: ((0, [2011]), (1, [2012]), (2, [2013]), (3, [2014]), (4, [2015]), (5, [2016]), (6, [2017]), (7, [2018]), (8, [2019]), (9, [2020]), (10, [2021]), (11, [2022]), (12, [2023]), (13, [2024]), (14, [2025])), subticks: none),
      ylim: (0, 105.0),
      lq.bar(xs.map(x => x - 0.28), (15.00, 1.00, 1.04, 2.04, 9.89, 8.42, 3.26, 2.22, 6.74, 10.64, 1.10, 2.17, 3.33, 1.19, 1.30,), width: 0.25, fill: red.lighten(20%), label: [same instance (.mzn + .dzn)]),
      lq.bar(xs.map(x => x), (70.83, 45.83, 46.43, 40.62, 48.15, 73.91, 62.50, 95.00, 48.15, 43.48, 25.93, 60.00, 45.00, 36.36, 42.11,), width: 0.25, fill: orange.lighten(20%), label: [same .mzn]),
      lq.bar(xs.map(x => x + 0.28), (90.00, 68.42, 70.00, 75.00, 84.21, 89.47, 75.00, 95.00, 73.68, 57.89, 52.63, 65.00, 45.00, 47.37, 47.37,), width: 0.25, fill: blue.lighten(20%), label: [same problem family]),
    )
  },
  caption: [Leave-one-year-out leakage of the cpsat8/k1 training data: for each year $y$ (x-axis), the bars show the fraction of $y$'s instances whose identity also appears somewhere in the remaining 14 training years. Three identity levels are shown: same exact instance (.mzn + .dzn together), same MiniZinc model file (.mzn) but possibly different data, and same problem family (folder). The strictest level is the only one that is true data leakage; the looser levels reflect the MiniZinc Challenge's recurring problem structure that the AI is expected to generalise across. The years 2023, 2024, and 2025 are off-limits as evaluation years because the k1 and ek1 portfolios were selected based on those years; the comparable leakage profile of 2020-2022 makes them legitimate stand-ins.],
  ) <fig:year-leakage>
