#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    table.header(
      [AI], [Year], [n cpsat-picked], [Borda AI], [Borda cpsat8], [Norm.],
    ),
    [svc-k1], [2020], [80], [35.61], [37.39], [0.4451],
    [svc-k1], [2021], [80], [32.15], [43.85], [0.4019],
    [svc-k1], [2022], [72], [37.43], [34.57], [0.5199],
    [svc-k1], [*total*], [232], [105.19], [115.81], [0.4534],
    table.hline(),
    [svc-ek1], [2020], [82], [33.79], [41.21], [0.4121],
    [svc-ek1], [2021], [87], [31.19], [51.81], [0.3585],
    [svc-ek1], [2022], [84], [45.14], [38.86], [0.5373],
    [svc-ek1], [*total*], [253], [110.12], [131.88], [0.4352],
  ),
  caption: [Head-to-head Borda restricted to instances where the AI predicted cpsat. On these instances the AI's executed schedule is functionally identical to the cpsat8 baseline except for the seconds spent on the no-yuck-equivalent static portfolio while features are being extracted, so any shortfall here measures the feature-extraction overhead rather than a wrong portfolio choice. Per-instance maximum is 1.],
  ) <tab:vs-cpsat-when-cpsat>
