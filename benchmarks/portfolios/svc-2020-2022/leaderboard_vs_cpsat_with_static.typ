#figure(
  table(
    columns: 5,
    align: (left, left, right, right, right),
    table.header(
      [Submission], [Type], [Borda], [Borda cpsat8], [n],
    ),
    [svc-k1], [AI], [129.25], [132.75], [285],
    [k1-8c-8s-v1], [static alt], [104.05], [161.95], [285],
    table.hline(),
    [svc-ek1], [AI], [123.27], [138.73], [285],
    [ek1-8c-8s-v2], [static alt], [90.51], [175.49], [285],
  ),
  caption: [Head-to-head Borda against the #cpsat() baseline on the 2020-2022 evaluation set, for each AI selector and its corresponding static alternative portfolio. Comparing the AI row to the static-alt row in each track shows whether the AI's selection beats running the alt portfolio unconditionally.],
  ) <tab:vs-cpsat-with-static>
