#figure(
  table(
    columns: 5,
    align: (left, left, right, right, right),
    table.header(
      [AI], [Static alternative], [Borda score AI], [Borda score static], [n],
    ),
    [svc-k1], [k1-8c-8s-v1], [162.00], [104.00], [285],
    [svc-ek1], [ek1-8c-8s-v2], [169.39], [96.61], [285],
  ),
  caption: [Direct head-to-head Borda score between each AI selector and its corresponding static alternative portfolio on the 2020-2022 evaluation set. Each instance is scored pairwise: the better solver wins outright, otherwise the point is split by wall-clock time.],
  ) <tab:ai-vs-static>
