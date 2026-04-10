#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s")
#let scores = (1761.22, 1625.27, 1645.60, 1724.57, 1732.85, 1731.96, 1701.06)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score: best-static Bound-Sharing (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (1598.1, 1788.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
