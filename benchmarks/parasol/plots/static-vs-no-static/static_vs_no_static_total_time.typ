#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("static", "no-static")
#let times = (40364.7, 40492.7)

#lq.diagram(
  width: 8cm,
  height: 6cm,
  title: [Total Solving Time: Static vs.\ No-Static],
  ylabel: [Total time (s)],
  ylim: (40108.7, 40748.7),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), times,
    fill: blue.lighten(30%),
  ),
)
