#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt,
    table-header([Method], [3-way Borda], [Ratio to Oracle]),
    [always-cpsat], [1638.11], [0.845],
    [always-k1], [1322.94], [0.683],
    [always-ek1], [1123.95], [0.580],
    [AI-k1 deployment], [1721.75], [0.889],
    [AI-ek1 deployment], [1681.18], [0.868],
    [Oracle (best of cpsat/k1/ek1)], [1937.48], [1.000],
  ),
  caption: [Total LOYO Borda across all 15 years, scored against the 3-way tournament of {cpsat, k1, ek1}. Each AI's prediction maps to a portfolio choice; that portfolio's 3-way Borda on the instance is added to the AI's total. The two AI deployments are evaluated on the same scale and are therefore directly comparable.],
) <tab:ai-vs-ai-totals>
