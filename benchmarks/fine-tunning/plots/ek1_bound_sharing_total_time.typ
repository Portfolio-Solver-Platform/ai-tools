#import "@preview/lilaq:0.6.0" as lq

#set page(width: auto, height: auto, margin: 1em)

#let labels = ("2s", "4s", "8s", "16s", "32s", "64s", "96s", "128s", "160s", "192s", "224s", "256s")
#let deltas = (-420.8, 294.6, 35.7, -230.4, -531.1, -321.5, -1452.9, 991.3, -2309.8, -2047.7, -235.9, 483.0)

#lq.diagram(
  width: 14cm,
  height: 6cm,
  title: [Time Saved vs.\ No Bound Sharing (ek1-8c-8s-v2, baseline: 45,659s)],
  ylabel: [Time saved (s)],
  xlabel: [Bound-sharing interval],
  xaxis: (
    ticks: labels.enumerate(),
    subticks: none,
  ),

  lq.plot((0, 11), (0, 0), stroke: (dash: "dashed", paint: black)),

  lq.plot(
    range(labels.len()), deltas,
    mark: "o",
    mark-size: 8pt,
    stroke: 1.5pt,
  ),
)
