#figure(
  table(
    columns: 6,
    align: (left, left, right, right, right, right),
    table.header(
      [AI], [AI picked], [n], [Borda AI], [Borda cpsat8], [#sym.Delta],
    ),
    [svc-k1], [cpsat (overhead)], [232], [105.19], [115.81], [-10.62],
    [svc-k1], [alt (decision)], [45], [23.56], [16.44], [+7.12],
    [svc-k1], [*combined*], [277], [128.75], [132.25], [-3.50],
    table.hline(),
    [svc-ek1], [cpsat (overhead)], [253], [110.12], [131.88], [-21.76],
    [svc-ek1], [alt (decision)], [24], [12.65], [6.35], [+6.30],
    [svc-ek1], [*combined*], [277], [122.77], [138.23], [-15.46],
  ),
  caption: [Decomposition of each AI selector's Borda gap against the `cpsat8` baseline over 2020-2022, split by what the AI predicted on each instance. The `cpsat (overhead)` row is where the AI's selection algorithmically matched the baseline -- any negative #sym.Delta there comes purely from the feature-extraction wall-clock cost the AI pays before it can hand cpsat the cores. The `alt (decision)` row is where the AI deviated from cpsat by selecting the k1/ek1 portfolio; its #sym.Delta isolates the decision quality of the AI's portfolio choice. The `combined` row is the union and matches the head-to-head vs-cpsat number for these years. The reading: nearly all of the Borda gap is overhead; the AI's portfolio decisions are essentially neutral (svc-k1) or mildly profitable (svc-ek1).],
  ) <tab:decision-vs-overhead>
