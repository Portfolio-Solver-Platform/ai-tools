#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: 0.5pt,
    table-header([Preprocessing], [Plain SVC], [Bagged SVC-MW]),
    [StandardScaler], [*833.09*], [829.22],
    [QuantileTransformer], [824.02], [839.52],
    [signed-log + StandardScaler], [818.65], [*846.31*],
    [signed-log + QuantileTransformer], [822.25], [837.29],
    [RobustScaler], [820.67], [835.54],
    [PowerTransformer], [824.12], [839.97],
    [signed-log + RobustScaler], [814.33], [835.54],
    [signed-log + PowerTransformer], [823.15], [839.97],
  ),
  caption: [LOYO Borda by preprocessing on cpsat8_k1, for plain RBF SVC and bagged margin-weighted SVC. The best per column is in bold. Signed-log + StandardScaler is the *worst* preprocessing for a plain SVC but becomes the *best* once we add margin weighting and bagging - the preprocessing ranking flips when the model class changes.],
) <tab:preprocessing-full>
