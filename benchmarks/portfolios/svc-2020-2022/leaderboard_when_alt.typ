#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    table.header(
      [AI], [Year], [n alt-picked], [Borda AI], [Borda cpsat8], [Norm.],
    ),
    [svc-k1], [2020], [14], [8.81], [2.19], [0.6295],
    [svc-k1], [2021], [11], [2.87], [6.13], [0.2613],
    [svc-k1], [2022], [20], [11.88], [8.12], [0.5938],
    [svc-k1], [*total*], [45], [23.56], [16.44], [0.5236],
    table.hline(),
    [svc-ek1], [2020], [12], [5.38], [3.62], [0.4487],
    [svc-ek1], [2021], [4], [1.41], [0.59], [0.3529],
    [svc-ek1], [2022], [8], [5.85], [2.15], [0.7316],
    [svc-ek1], [*total*], [24], [12.65], [6.35], [0.5270],
  ),
  caption: [Head-to-head Borda restricted to instances where the AI *deviated* from cpsat by picking the alt portfolio (k1-8c-8s-v1 for svc-k1, ek1-8c-8s-v2 for svc-ek1). These rows measure the *decision quality* of the AI's non-cpsat picks: a normalised score above 0.5 means the alt portfolio beat cpsat on the instances the AI chose to deviate on, below 0.5 means the deviation cost Borda.],
  ) <tab:vs-cpsat-when-alt>
