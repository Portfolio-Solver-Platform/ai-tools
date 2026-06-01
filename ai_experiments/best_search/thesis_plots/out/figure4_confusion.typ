#figure(
  table(
    columns: 3,
    align: (left+horizon, center+horizon, center+horizon),
    stroke: 0.5pt,
    table-header([], [Predicted cpsat], [Predicted k1]),
    [True cpsat], [938], [75],
    [True k1],    [256], [110],
  ),
  caption: [Out-of-fold confusion matrix on cpsat8_k1 (15-fold LOYO, 1379 test instances). Precision on the k1 class is #text(weight: "bold")[59.5%] at recall #text(weight: "bold")[30.1%]. The model is conservative on k1 - it only commits to a k1 prediction when reasonably confident, trading recall for precision to avoid borda-losing wrong picks.],
) <tab:confusion-k1>
