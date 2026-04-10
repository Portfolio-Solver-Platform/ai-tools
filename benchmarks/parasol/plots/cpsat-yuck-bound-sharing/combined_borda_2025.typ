#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s")
#let scores = (1496.82, 1511.88, 1537.78, 1519.95, 1554.81, 1541.97, 1516.51)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (vs.\ All Solvers)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (1485.2, 1566.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
