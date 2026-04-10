#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("none", "2s", "4s", "8s", "16s", "32s", "64s")
#let scores = (266.94, 278.95, 290.88, 287.79, 303.31, 298.86, 279.27)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Borda Score by Bound-Sharing Interval],
  ylabel: [Borda score],
  xlabel: [Bound-sharing interval],
  ylim: (259.7, 310.6),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), scores,
    fill: blue.lighten(30%),
  ),
)
