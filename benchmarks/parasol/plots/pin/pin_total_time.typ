#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("pin", "no-pin")
#let times = (59400.7, 59433.6)

#lq.diagram(
  width: 8cm,
  height: 6cm,
  title: [Total Solving Time: Pin vs.\ No-Pin],
  ylabel: [Total time (s)],
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.bar(
    range(labels.len()), times,
    fill: blue.lighten(30%),
  ),
)
