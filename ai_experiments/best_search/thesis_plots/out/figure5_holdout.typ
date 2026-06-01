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
        xlabel: [Method], ylabel: [Test Borda (80/20 holdout)],
        xaxis: (ticks: ((0, [always-cpsat]), (1, [always-other]), (2, [random]), (3, [LogReg]), (4, [Plain SVC]), (5, [BagSVC-MW])), subticks: none),
        lq.bar(xs, (158.74, 114.46, 131.18, 162.91, 165.84, 172.48), fill: blue.lighten(20%)),
        lq.plot(xs, (158.74, 114.46, 131.18, 162.91, 165.84, 172.48), yerr: (1.58, 1.65, 3.41, 3.30, 1.52, 3.94), stroke: none, color: black),
      )
    },
    {
      let n = 6
      let xs = range(n).map(i => float(i))
      lq.diagram(
        width: 14cm, height: 5cm,
        title: [cpsat8 vs ek1],
        xlabel: [Method], ylabel: [Test Borda (80/20 holdout)],
        xaxis: (ticks: ((0, [always-cpsat]), (1, [always-other]), (2, [random]), (3, [LogReg]), (4, [Plain SVC]), (5, [BagSVC-MW])), subticks: none),
        lq.bar(xs, (168.38, 103.62, 136.67, 169.69, 170.20, 178.49), fill: blue.lighten(20%)),
        lq.plot(xs, (168.38, 103.62, 136.67, 169.69, 170.20, 178.49), yerr: (2.30, 1.80, 3.57, 1.28, 2.25, 2.98), stroke: none, color: black),
      )
    }
  ),
  caption: [80/20 stratified random-split holdout, averaged over 5 random seeds. All methods use FIXED hyperparameters (no HPO at evaluation time); BagSVC-MW uses the median LOYO params from the cpsat8_k1 sweep applied unchanged to both datasets. Error bars are ± one standard deviation across seeds. The 80/20 ratios are higher than the LOYO ratios because year-stratification (the LOYO setup) is a strictly harder verification than random splits.],
) <fig:holdout>
