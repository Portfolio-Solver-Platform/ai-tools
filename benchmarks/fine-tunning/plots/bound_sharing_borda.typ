#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let scores = (649.30, 599.83, 622.00, 648.45, 548.68, 569.68, 623.83, 571.24, 572.54, 642.35, 577.25, 567.09, 607.78)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval (k1-8c-8s-v1)],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (528.6, 669.4),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
