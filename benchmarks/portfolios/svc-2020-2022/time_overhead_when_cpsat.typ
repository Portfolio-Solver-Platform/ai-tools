#figure(
  table(
    columns: 7,
    align: (left, right, right, right, right, right, right),
    table.header(
      [AI], [n], [med (s)], [med slowdown (s)], [mean slowdown (s)], [p90 (s)], [AI slower],
    ),
    [svc-k1], [231], [0.68], [0.89], [2.92], [1.95], [70.6%],
    [svc-ek1], [252], [0.76], [1.03], [2.72], [2.02], [66.7%],
  ),
  caption: [Wall-clock difference $text("AI") - text("cpsat8")$ on instances where the AI predicted cpsat. *med (s)* is the median over all such instances; the two slowdown columns are the median and mean *restricted to instances where the AI was actually slower than the cpsat baseline*, since the overall mean is dragged below zero by a handful of long instances where the static portfolio solved the problem before cpsat would have. *AI slower* is the share of instances where the difference is strictly positive.],
  ) <tab:time-overhead-when-cpsat>
