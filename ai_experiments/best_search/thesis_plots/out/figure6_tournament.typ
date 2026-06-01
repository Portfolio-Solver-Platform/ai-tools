#figure(
  {
    let n = 3
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 11cm, height: 6cm,
      title: [3-way LOYO Borda tournament],
      xlabel: [Submission], ylabel: [Total Borda (out of 2758)],
      xaxis: (ticks: ((0, [cpsat]), (1, [svc_k1]), (2, [svc_ek1])), subticks: none),
      ylim: (1259, 1419),
      lq.bar(xs, (1309.52, 1393.42, 1357.06,), fill: blue.lighten(20%)),
    )
  },
  caption: [Simulated MiniZinc Challenge among the three deployable submissions, scored by 3-way pairwise Borda using the MZN Challenge rules (see @minizinc_challenge). On every test instance each submission is compared head-to-head against the other two; each pairwise outcome lies in $[0, 1]$ (the better solution wins outright, equally-good solutions split the point in proportion to wall-clock time so the faster solver gets more). The per-instance score is the sum of the two pairwise outcomes, and the maximum total per submission is 2758 = 2 #sym.times 1379 instances; the y-axis is zoomed to make the gap between submissions readable. Predictions are leave-one-year-out: for each instance, the SVC's prediction comes from the model trained on the other 14 years.],
  ) <fig:three-way-tournament>

#figure(
  {
    let n = 2
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 11cm, height: 6cm,
      title: [Head-to-head: svc_k1 vs cpsat],
      xlabel: [Submission], ylabel: [Total Borda (out of 1379)],
      xaxis: (ticks: ((0, [cpsat]), (1, [svc_k1])), subticks: none),
      ylim: (617, 719),
      lq.bar(xs, (649.04, 702.96,), fill: blue.lighten(20%)),
    )
  },
  caption: [Head-to-head MZN-Challenge Borda between svc_k1 and the always-cpsat baseline, scored under the MZN Challenge rules (see @minizinc_challenge). On every instance the two submissions are compared once: the better solution wins outright, equally-good solutions split the point in proportion to wall-clock time so the faster solver gets more. The maximum total per submission is 1379 = 1 #sym.times 1379 instances; the y-axis is zoomed to make the gap readable. Predictions are leave-one-year-out.],
  ) <fig:h2h-svc-k1-vs-cpsat>

#figure(
  {
    let n = 2
    let xs = range(n).map(i => float(i))
    lq.diagram(
      width: 11cm, height: 6cm,
      title: [Head-to-head: svc_ek1 vs cpsat],
      xlabel: [Submission], ylabel: [Total Borda (out of 1379)],
      xaxis: (ticks: ((0, [cpsat]), (1, [svc_ek1])), subticks: none),
      ylim: (641, 703),
      lq.bar(xs, (660.48, 693.52,), fill: blue.lighten(20%)),
    )
  },
  caption: [Head-to-head MZN-Challenge Borda between svc_ek1 and the always-cpsat baseline, scored under the MZN Challenge rules (see @minizinc_challenge). On every instance the two submissions are compared once: the better solution wins outright, equally-good solutions split the point in proportion to wall-clock time so the faster solver gets more. The maximum total per submission is 1379 = 1 #sym.times 1379 instances; the y-axis is zoomed to make the gap readable. Predictions are leave-one-year-out.],
  ) <fig:h2h-svc-ek1-vs-cpsat>
