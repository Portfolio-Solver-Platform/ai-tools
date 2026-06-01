#figure(
  grid(
    columns: 1,
    row-gutter: 0.8em,
    {
      let n = 6
      let xs = range(n).map(i => float(i))
      lq.diagram(
        width: 14cm, height: 5cm,
        title: [cpsat8 vs k1],
        xlabel: [Method], ylabel: [Total Borda],
        xaxis: (ticks: ((0, [always-other]), (1, [random]), (2, [always-cpsat]), (3, [Plain SVC]), (4, [BagSVC-MW]), (5, [Oracle])), subticks: none),
        lq.bar(xs, (574.13, 681.54, 788.87, 833.09, 846.31, 966.61), fill: blue.lighten(20%)),
      )
    },
    {
      let n = 6
      let xs = range(n).map(i => float(i))
      lq.diagram(
        width: 14cm, height: 5cm,
        title: [cpsat8 vs ek1],
        xlabel: [Method], ylabel: [Total Borda],
        xaxis: (ticks: ((0, [always-other]), (1, [random]), (2, [always-cpsat]), (3, [Plain SVC]), (4, [BagSVC-MW]), (5, [Oracle])), subticks: none),
        lq.bar(xs, (514.76, 681.08, 849.24, 860.04, 883.54, 986.09), fill: blue.lighten(20%)),
      )
    },
  ),
  caption: [Total LOYO Borda across baselines and the final BagSVC-MW model, for both portfolio decisions. "always-other" picks k1 (left) or ek1 (right) on every instance; "random" is a uniform coin flip averaged over 50 seeds; "always-cpsat" is the no-AI fallback. The final model beats every simple baseline and lies close to the per-instance oracle.],
) <fig:total-comparison>
