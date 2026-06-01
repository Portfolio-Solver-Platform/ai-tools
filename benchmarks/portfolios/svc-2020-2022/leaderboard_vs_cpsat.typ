#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    table.header(
      [AI], [Year], [Borda AI], [Borda cpsat8], [Norm.], [n],
    ),
    [svc-k1], [2020], [44.42], [39.58], [0.4725], [94],
    [svc-k1], [2021], [35.02], [49.98], [0.3648], [96],
    [svc-k1], [2022], [49.81], [43.19], [0.5243], [95],
    [svc-k1], [*total*], [129.25], [132.75], [0.4535], [285],
    table.hline(),
    [svc-ek1], [2020], [39.18], [44.82], [0.4168], [94],
    [svc-ek1], [2021], [32.60], [52.40], [0.3396], [96],
    [svc-ek1], [2022], [51.49], [41.51], [0.5420], [95],
    [svc-ek1], [*total*], [123.27], [138.73], [0.4325], [285],
  ),
  caption: [Head-to-head MZN-Challenge Borda of each AI selector (`svc-k1`, `svc-ek1`) against the `cpsat8` baseline, broken down by held-out year. Each row uses the intersection of instances both submissions ran. The per-instance maximum is 1 (winner takes the full point, ties split by wall-clock time), so $text("Borda AI") + text("Borda cpsat8") = n$ in every row. Normalised column is $text("Borda AI") / n$.],
  ) <tab:vs-cpsat>
