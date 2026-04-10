#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("choco-cpsat", "choco-highs")
#let pin-times = (60554.7, 82032.4)
#let not-pin-times = (59357.7, 81840.0)

#lq.diagram(
  width: 10cm,
  height: 6cm,
  title: [Total Solving Time: Pin vs.\ No-Pin],
  ylabel: [Total time (s)],
  xlabel: [Portfolio],
  legend: (position: (100% + .5em, 0%)),
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), pin-times,
    offset: -0.2, width: 0.4,
    fill: blue.lighten(30%),
    label: [pin],
  ),
  lq.bar(
    range(labels.len()), not-pin-times,
    offset: 0.2, width: 0.4,
    fill: orange.lighten(30%),
    label: [not-pin],
  ),
)
