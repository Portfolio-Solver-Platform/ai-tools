#figure(
  table(
    columns: 7,
    align: (left+horizon, right, right, right, right, right, right+horizon),
    stroke: 0.5pt,
    table-header([Method], [Borda], [Ratio], [Acc], [Borda], [Ratio], [Acc]),
    table-header([], table.cell(colspan: 3)[#text(weight: "bold")[cpsat8 vs k1]], table.cell(colspan: 3)[#text(weight: "bold")[cpsat8 vs ek1]]),
    [always-cpsat], [788.87], [0.816], [73.5%], [849.24], [0.861], [80.2%],
    [Plain SVC], [833.09], [0.862], [77.0%], [860.04], [0.872], [79.8%],
    [BagSVC-MW], [846.31], [0.876], [76.4%], [883.54], [0.896], [81.1%],
    [Oracle], [966.61], [1.000], [---], [986.09], [1.000], [---],
  ),
  caption: [Headline LOYO results on the two binary portfolio decisions. Borda is the total over the 1379 test instances across all 15 leave-one-year-out folds. Ratio is Borda divided by the oracle's per-instance maximum sum. Accuracy is the fraction of instances where the model's chosen class matches the winning class; the always-cpsat row reports the cpsat-wins rate. The oracle is the per-instance best.],
) <tab:headline>
